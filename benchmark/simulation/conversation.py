#!/usr/bin/env python
"""
Conversation Runner - Run multi-turn student-tutor conversations

Supports two modes:
  1. Interactive: Human plays the tutor, types responses in terminal
  2. Auto: Tutor backend (deep_tutor or mock)

Usage:
  # Interactive mode (editor by default; use --inline for console input):
  python -m benchmark.simulation.conversation --entry path/to/entry.json

  # Console input (empty line + Enter to send; paste may truncate long content):
  python -m benchmark.simulation.conversation --entry path/to/entry.json --inline

  # Auto mode with DeepTutor backend:
  python -m benchmark.simulation.conversation --entry path/to/entry.json --auto --auto-backend deep_tutor
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Literal

from benchmark.simulation.student_agent import StudentAgent

# Delimiter to separate student context from tutor response in editor
_EDITOR_DELIMITER = "\n\n--- Type your response below this line ---\n\n"

logger = logging.getLogger("benchmark.conversation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _suppress_noisy_auto_logs() -> None:
    """Suppress verbose INFO/DEBUG logs from RAG/LLM internals during simulation."""
    # Clamp root logger first to suppress generic "INFO: Process ..." style logs.
    logging.getLogger().setLevel(logging.WARNING)

    noisy_loggers = [
        "CodeExecutor",
        "RAGService",
        "RAGForward",
        "Main",
        "LLMClient",
        "EmbeddingClient",
        "lightrag",
        "raganything",
        "nano-vectordb",
        "multiprocessing",
        "openai",
        "httpx",
        "httpcore",
        "src.services.embedding.provider",
        "src.services.embedding.adapters.openai_compatible",
        "src.services.rag",
        "src.services.rag.service",
        "src.tools.rag_tool",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def _get_tutor_input_via_editor(student_context: str) -> str | None:
    """
    Open editor with student's message at top for reference.
    Returns tutor response (content below delimiter), or None if aborted.
    Avoids terminal paste truncation (4096 bytes on Linux, ~16K on macOS).
    """
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    # Wrap long lines so they fit in editor (72 chars per line)
    wrapped_lines = []
    for para in student_context.strip().split("\n"):
        for wline in textwrap.wrap(para, width=72):
            wrapped_lines.append(f"# {wline}")
        wrapped_lines.append("#")  # blank line between paragraphs
    header = "# Student's question (read-only reference):\n#\n"
    header += "\n".join(wrapped_lines) + "\n"
    header += _EDITOR_DELIMITER

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(header)
        f.flush()
        path = f.name

    try:
        ret = subprocess.call([editor, path])
        if ret != 0:
            return None
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if _EDITOR_DELIMITER in text:
            result = text.split(_EDITOR_DELIMITER, 1)[1].strip()
        else:
            result = text.strip()
        if not result or result.lower() == "quit":
            return None
        return result
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# Fixed prompt sent to tutor after student says task_complete
TUTOR_POST_COMPLETE_PROMPT = (
    "The student has indicated they are done with this session. "
    "Based on this conversation, please create a practice problem that reinforces what was covered. "
    "Give only the problem statement (no solution)."
)


def _build_sim_workspace(profile_id: str | None, entry_id: str, shared_by_profile: bool) -> str:
    """Build workspace path for DeepTutor tool-based auto tutor."""
    base = PROJECT_ROOT / "benchmark" / "data" / "sim_workspaces"
    key = profile_id if (shared_by_profile and profile_id) else entry_id
    return str(base / key)


# Simple tutor system prompt for auto mode
MOCK_TUTOR_SYSTEM = (
    "You are a helpful, patient, and knowledgeable tutor. "
    "A student is asking you for help. Respond clearly and helpfully. "
    "If they ask for an explanation or walkthrough, provide it. "
    "If they ask for a practice problem, give them a similar problem (clear statement, no solution yet). "
    "Ask follow-up questions to check understanding. "
    "If the student shows a misconception, gently guide them toward the correct understanding "
    "rather than just stating the answer. Use examples when helpful. "
    "Keep responses focused and not too long (2-5 sentences typically)."
)


async def mock_tutor_respond(
    student_message: str,
    history: list[dict[str, str]],
    kb_name: str | None = None,
    prior_sessions_summary: str | None = None,
) -> str:
    """
    Generate a mock tutor response using LLM.

    Args:
        student_message: The student's latest message
        history: Conversation history from tutor's perspective
            (role="user" for student, role="assistant" for tutor)
        kb_name: Knowledge base name used for RAG retrieval
        prior_sessions_summary: Optional summary of previous sessions (tutor remembers)

    Returns:
        Tutor response string
    """
    from src.services.llm import factory
    from src.tools.rag_tool import rag_search

    system = MOCK_TUTOR_SYSTEM
    if prior_sessions_summary:
        system = (
            "You have had previous tutoring sessions with this student. "
            "Use this context to personalize your response.\n\n"
            "## Previous sessions\n"
            f"{prior_sessions_summary}\n\n"
            "---\n\n"
            + system
        )

    # Auto mode RAG: use raw student message as query in naive mode.
    rag_context = ""
    if kb_name:
        try:
            # Some RAG internals print INFO lines directly to stdout/stderr.
            # Silence those noisy streams during retrieval in auto mode.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rag_result = await rag_search(
                    query=student_message,
                    kb_name=kb_name,
                    mode="naive",
                )
            rag_answer = (rag_result.get("answer") or rag_result.get("content") or "").strip()
            if rag_answer:
                rag_context = (
                    "## Retrieved context (RAG, naive mode)\n"
                    f"{rag_answer[:4000]}\n\n"
                )
        except Exception as e:
            logger.warning("RAG retrieval failed for kb=%s: %s", kb_name, e)

    # No action hint; tutor infers intent from message content.
    user_content = f"{rag_context}## Student message\n{student_message}"

    # Build messages from tutor's perspective
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    response = await factory.complete(
        prompt="",
        system_prompt="",
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
    )

    return response


async def deep_tutor_respond(
    student_message: str,
    kb_name: str,
    workspace: str,
    language: str = "en",
) -> str:
    """
    Generate tutor response via DeepTutor solve pipeline.
    Uses student's raw message as question.
    """
    from benchmark.simulation.tools import solve_question

    if not kb_name:
        return "(DeepTutor unavailable: missing kb_name in entry.)"

    result = await solve_question(
        workspace=workspace,
        kb_name=kb_name,
        question=student_message,
        language=language,
    )
    answer = (result.get("answer") or "").strip()
    return answer or "(No answer generated.)"


def _format_question_block(q: dict) -> str:
    """Format one generated question for tutor output."""
    title = q.get("question", "").strip()
    options = q.get("options", {}) or {}
    lines = [title] if title else ["(Empty question)"]
    if isinstance(options, dict):
        for k, v in options.items():
            lines.append(f"{k}. {v}")
    elif isinstance(options, list):
        for item in options:
            lines.append(str(item))
    return "\n".join(lines)


async def deep_tutor_generate_practice_problem(
    *,
    kb_name: str,
    workspace: str,
    topic: str,
    language: str = "en",
) -> str:
    """
    Generate one practice problem via DeepTutor question pipeline.
    """
    from benchmark.simulation.tools import generate_questions

    if not kb_name:
        return "(Practice problem generation unavailable: missing kb_name in entry.)"

    result = await generate_questions(
        workspace=workspace,
        kb_name=kb_name,
        topic=topic,
        num_questions=1,
        language=language,
    )
    questions = result.get("questions", []) or []
    if not questions:
        return "(Practice problem generation failed.)"
    return _format_question_block(questions[0])


def _summarize_session(transcript: list[dict], task: dict, session_index: int) -> str:
    """Produce a brief summary of a session for prior_sessions context."""
    title = task.get("title", "Unknown task")
    lines = [f"Session {session_index}: {title}"]
    # First exchange
    for i, msg in enumerate(transcript[:4]):
        role = msg.get("role", "?")
        content = (msg.get("content", "") or "").strip()[:150]
        if content:
            lines.append(f"  {role}: {content}...")
    lines.append(f"  ({len(transcript)} messages total)")
    return "\n".join(lines)


async def _run_single_session(
    entry: dict,
    max_turns: int,
    auto: bool,
    use_editor: bool,
    auto_backend: Literal["deep_tutor", "mock"] = "deep_tutor",
    deeptutor_workspace: str | None = None,
    deeptutor_language: str = "en",
    prior_sessions_summary: str | None = None,
) -> dict:
    """
    Run one session (one task). Returns result dict with transcript, entry, etc.
    """
    agent = StudentAgent.from_entry(
        entry,
        prior_sessions_context=prior_sessions_summary,
    )
    entry_id = entry.get("entry_id", "unknown")
    kb_name = entry.get("kb_name")
    profile_id = entry.get("profile", {}).get("profile_id")
    workspace = deeptutor_workspace or _build_sim_workspace(
        profile_id=profile_id,
        entry_id=entry_id,
        shared_by_profile=True,
    )
    task_title = entry.get("task", {}).get("title", "")

    tutor_history: list[dict[str, str]] = []
    student_msg = agent.initial_message()

    print(f"[Student] {student_msg}\n")

    for turn in range(1, max_turns):
        if auto:
            try:
                if auto_backend == "deep_tutor":
                    tutor_msg = await deep_tutor_respond(
                        student_message=student_msg,
                        kb_name=kb_name,
                        workspace=workspace,
                        language=deeptutor_language,
                    )
                else:
                    tutor_msg = await mock_tutor_respond(
                        student_msg,
                        tutor_history,
                        kb_name=kb_name,
                        prior_sessions_summary=prior_sessions_summary,
                    )
            except Exception as e:
                logger.error("Auto tutor failed (%s): %s", auto_backend, e)
                print(f"\n[Tutor] Error: {e}")
                break
        else:
            if use_editor:
                tutor_msg = _get_tutor_input_via_editor(student_msg)
            else:
                print("[Tutor] (type response, empty line + Enter to send, 'quit' to end)")
                lines = []
                quit_typed = False
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line.strip().lower() == "quit":
                        quit_typed = True
                        break
                    if line == "" and lines:
                        break
                    lines.append(line)
                if quit_typed or not lines:
                    break
                tutor_msg = "\n".join(lines)

            if tutor_msg is None:
                break

        tutor_history.append({"role": "user", "content": student_msg})
        tutor_history.append({"role": "assistant", "content": tutor_msg})

        print(f"[Tutor] {tutor_msg}\n")

        student_msg, student_action = await agent.respond(tutor_msg)
        print(f"[Student] {student_msg}\n")
        if student_action == "task_complete":
            print("[System] Student indicated task complete. Requesting practice problem from tutor...")
            tutor_history.append({"role": "user", "content": student_msg})
            if auto:
                try:
                    if auto_backend == "deep_tutor":
                        topic = (
                            f"{task_title}\n"
                            f"Conversation summary request: {TUTOR_POST_COMPLETE_PROMPT}\n"
                            "Generate one practice problem aligned to this session."
                        )
                        practice_msg = await deep_tutor_generate_practice_problem(
                            kb_name=kb_name,
                            workspace=workspace,
                            topic=topic,
                            language=deeptutor_language,
                        )
                    else:
                        practice_msg = await mock_tutor_respond(
                            TUTOR_POST_COMPLETE_PROMPT,
                            tutor_history,
                            kb_name=kb_name,
                            prior_sessions_summary=prior_sessions_summary,
                        )
                except Exception as e:
                    logger.error("Auto tutor post-complete failed (%s): %s", auto_backend, e)
                    practice_msg = "(Practice problem generation failed.)"
            else:
                print(f"[Tutor] {TUTOR_POST_COMPLETE_PROMPT}")
                print("[Tutor] (type practice problem, empty line + Enter to send)")
                lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line == "" and lines:
                        break
                    lines.append(line)
                practice_msg = "\n".join(lines) if lines else "(No practice problem provided.)"
            tutor_history.append({"role": "assistant", "content": practice_msg})
            print(f"[Tutor] {practice_msg}\n")
            # Append to agent history so it appears in transcript
            agent.history.append({"role": "user", "content": practice_msg})
            break

    transcript = agent.get_transcript()
    return {
        "entry_id": entry_id,
        "transcript": transcript,
        "entry": entry,
        "actual_turns": agent.turn_count,
    }


async def run_conversation(
    entry_path: str | Path,
    max_turns: int = 20,
    auto: bool = False,
    auto_backend: Literal["deep_tutor", "mock"] = "deep_tutor",
    deeptutor_language: str = "en",
    output_dir: str | Path | None = None,
    entry_index: int = 0,
    use_editor: bool = True,
) -> dict:
    """
    Run a multi-turn conversation between student agent and tutor.

    Args:
        entry_path: Path to benchmark entry JSON file, or JSONL file
        max_turns: Maximum number of student turns (including initial message, default: 20)
        auto: If True, use mock LLM tutor. If False, interactive (stdin).
        output_dir: Directory to save transcript. If None, uses default.
        entry_index: When entry_path is JSONL, which entry to use (0-based).
        use_editor: If True (default), open $EDITOR. If False, use console input.

    Returns:
        Conversation result dict with transcript and metadata
    """
    entry_path = Path(entry_path)

    # Load entry and create agent
    with open(entry_path, encoding="utf-8") as f:
        if entry_path.suffix.lower() == ".jsonl":
            lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                raise ValueError(f"Empty JSONL file: {entry_path}")
            if entry_index >= len(lines):
                raise ValueError(
                    f"entry_index={entry_index} out of range (file has {len(lines)} entries)"
                )
            entry = json.loads(lines[entry_index])
        else:
            entry = json.load(f)

    agent = StudentAgent.from_entry(entry)
    entry_id = entry.get("entry_id", entry_path.stem)
    kb_name = entry.get("kb_name")
    profile_id = entry.get("profile", {}).get("profile_id")
    workspace = _build_sim_workspace(
        profile_id=profile_id,
        entry_id=entry_id,
        shared_by_profile=False,
    )
    task_title = entry.get("task", {}).get("title", "")

    print(f"\n{'='*60}")
    print(f"Conversation: {entry_id}")
    mode_desc = f"auto ({auto_backend})" if auto else "interactive (you are the tutor)"
    print(f"Mode: {mode_desc}")
    print(f"Max turns: {max_turns}")
    print(f"{'='*60}\n")

    # Tutor-side history (from tutor's perspective)
    tutor_history: list[dict[str, str]] = []

    # Turn 0: Student's initial message
    student_msg = agent.initial_message()
    print(f"[Student] {student_msg}\n")

    for turn in range(1, max_turns):
        # Get tutor response
        if auto:
            try:
                if auto_backend == "deep_tutor":
                    tutor_msg = await deep_tutor_respond(
                        student_message=student_msg,
                        kb_name=kb_name,
                        workspace=workspace,
                        language=deeptutor_language,
                    )
                else:
                    tutor_msg = await mock_tutor_respond(
                        student_msg, tutor_history, kb_name=kb_name
                    )
            except Exception as e:
                logger.error("Auto tutor failed (%s): %s", auto_backend, e)
                print(f"\n[Tutor] Error: {e}")
                print("(Conversation stopped. Partial transcript will be saved.)")
                break
        else:
            print("-" * 40)
            if use_editor:
                print("[Tutor] Opening editor (student's question at top).")
                print("        Save & close to send. nano: Ctrl+O then Ctrl+X | vim: :wq | VS Code: Cmd+S then close tab")
                tutor_msg = _get_tutor_input_via_editor(student_msg)
                if tutor_msg is None:
                    print("\n[Conversation ended by tutor]")
                    break
            else:
                print("[Tutor] (type your response, empty line + Enter to send, 'quit' to end)")
                lines = []
                quit_typed = False
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line.strip().lower() == "quit":
                        print("\n[Conversation ended by tutor]")
                        quit_typed = True
                        break
                    if line == "" and lines:
                        break
                    lines.append(line)
                if quit_typed or not lines:
                    break
                tutor_msg = "\n".join(lines)

        # Record in tutor history
        tutor_history.append({"role": "user", "content": student_msg})
        tutor_history.append({"role": "assistant", "content": tutor_msg})

        print(f"[Tutor] {tutor_msg}\n")

        # Get student response (task_complete only when ending)
        student_msg, student_action = await agent.respond(tutor_msg)
        print(f"[Student] {student_msg}\n")
        if student_action == "task_complete":
            print("[System] Student indicated task complete. Requesting practice problem from tutor...")
            tutor_history.append({"role": "user", "content": student_msg})
            if auto:
                try:
                    if auto_backend == "deep_tutor":
                        topic = (
                            f"{task_title}\n"
                            f"Conversation summary request: {TUTOR_POST_COMPLETE_PROMPT}\n"
                            "Generate one practice problem aligned to this session."
                        )
                        practice_msg = await deep_tutor_generate_practice_problem(
                            kb_name=kb_name,
                            workspace=workspace,
                            topic=topic,
                            language=deeptutor_language,
                        )
                    else:
                        practice_msg = await mock_tutor_respond(
                            TUTOR_POST_COMPLETE_PROMPT,
                            tutor_history,
                            kb_name=kb_name,
                        )
                except Exception as e:
                    logger.error("Auto tutor post-complete failed (%s): %s", auto_backend, e)
                    practice_msg = "(Practice problem generation failed.)"
            else:
                print(f"[Tutor] {TUTOR_POST_COMPLETE_PROMPT}")
                print("[Tutor] (type practice problem, empty line + Enter to send)")
                lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        break
                    if line == "" and lines:
                        break
                    lines.append(line)
                practice_msg = "\n".join(lines) if lines else "(No practice problem provided.)"
            print(f"[Tutor] {practice_msg}\n")
            agent.history.append({"role": "user", "content": practice_msg})
            break

    print(f"{'='*60}")
    print(f"Conversation complete. {agent.turn_count} student turns.")
    print(f"{'='*60}\n")

    # Build result
    result = {
        "entry_id": entry_id,
        "timestamp": datetime.now().isoformat(),
        "mode": "auto" if auto else "interactive",
        "max_turns": max_turns,
        "actual_turns": agent.turn_count,
        "transcript": agent.get_transcript(),
        "entry": entry,
    }

    # Save transcript
    if output_dir is None:
        output_dir = PROJECT_ROOT / "benchmark" / "data" / "transcripts"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{entry_id}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved → {output_file}")
    return result


def _load_entries_for_profile(
    jsonl_path: Path,
    profile_id: str,
) -> list[dict]:
    """Load and return entries for a given profile_id, sorted by task_id."""
    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("profile", {}).get("profile_id") == profile_id:
                entries.append(entry)
    # Sort by entry_id (contains task_id) for deterministic order
    entries.sort(key=lambda e: e.get("entry_id", ""))
    return entries


def _load_entries_from_paths(paths: list[str]) -> list[dict]:
    """Load entries from explicit JSON paths."""
    entries = []
    for p in paths:
        path = Path(p.strip())
        if not path.exists():
            raise FileNotFoundError(f"Entry file not found: {path}")
        with open(path, encoding="utf-8") as f:
            entries.append(json.load(f))
    return entries


async def run_multi_session(
    entry_path: str | Path | None = None,
    profile_id: str | None = None,
    entry_paths: list[str] | None = None,
    max_turns: int = 20,
    auto: bool = False,
    auto_backend: Literal["deep_tutor", "mock"] = "deep_tutor",
    deeptutor_language: str = "en",
    output_dir: str | Path | None = None,
    use_editor: bool = True,
    evolve_profile: bool = True,
) -> dict:
    """
    Run multiple sessions for the same student (one task per session).

    Entries are loaded either by:
    - profile_id: filter JSONL at entry_path by profile
    - entry_paths: explicit list of entry JSON paths

    After each session, prior_sessions_summary is built and injected into
    student and tutor prompts. If evolve_profile is True, the profile is
    evolved (resolved gaps → known_well) for the next session.

    Returns:
        Dict with sessions list and combined transcript
    """
    if entry_paths:
        entries = _load_entries_from_paths(entry_paths)
    elif entry_path and profile_id:
        entries = _load_entries_for_profile(Path(entry_path), profile_id)
        if not entries:
            raise ValueError(
                f"No entries found for profile_id={profile_id} in {entry_path}"
            )
    else:
        raise ValueError("Provide either (entry_path + profile_id) or entry_paths")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "benchmark" / "data" / "transcripts"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_id_display = profile_id or entries[0].get("profile", {}).get("profile_id", "?")
    shared_workspace = _build_sim_workspace(
        profile_id=profile_id_display,
        entry_id=entries[0].get("entry_id", "unknown"),
        shared_by_profile=True,
    )
    print(f"\n{'='*60}")
    print(f"Multi-session: {profile_id_display} ({len(entries)} sessions)")
    print(f"Mode: {'auto (' + auto_backend + ')' if auto else 'interactive'}")
    if auto and auto_backend == "deep_tutor":
        print(f"DeepTutor workspace(shared): {shared_workspace}")
    print(f"Evolve profile: {evolve_profile}")
    print(f"{'='*60}\n")

    prior_sessions_summary: list[str] = []
    current_profile = entries[0].get("profile", {})
    sessions_results: list[dict] = []

    for i, base_entry in enumerate(entries):
        session_num = i + 1
        entry_id = base_entry.get("entry_id", f"session_{session_num}")

        # Build entry for this session: evolved profile + prior context
        entry = dict(base_entry)
        if evolve_profile and i > 0:
            prev_entry = entries[i - 1]
            resolved = prev_entry.get("gaps", [])
            from benchmark.simulation.profile_evolver import evolve_profile as evolve_profile_fn

            current_profile = evolve_profile_fn(prev_entry.get("profile", {}), resolved)
        entry["profile"] = current_profile

        prior_ctx = "\n".join(prior_sessions_summary) if prior_sessions_summary else None

        print(f"\n--- Session {session_num}/{len(entries)}: {entry_id} ---\n")

        result = await _run_single_session(
            entry=entry,
            max_turns=max_turns,
            auto=auto,
            use_editor=use_editor,
            auto_backend=auto_backend,
            deeptutor_workspace=shared_workspace,
            deeptutor_language=deeptutor_language,
            prior_sessions_summary=prior_ctx,
        )

        task = entry.get("task", {})
        summary = _summarize_session(
            result["transcript"],
            task,
            session_num,
        )
        prior_sessions_summary.append(summary)

        sessions_results.append(result)
        print(f"\n[Session {session_num} complete. {result['actual_turns']} turns.]")

    # Save combined result
    combined = {
        "profile_id": profile_id_display,
        "timestamp": datetime.now().isoformat(),
        "mode": "auto" if auto else "interactive",
        "evolve_profile": evolve_profile,
        "num_sessions": len(sessions_results),
        "sessions": [
            {
                "entry_id": r["entry_id"],
                "actual_turns": r["actual_turns"],
                "transcript": r["transcript"],
                "entry": r["entry"],
            }
            for r in sessions_results
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"multi_{profile_id_display}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Multi-session complete. Transcript saved → {out_file}")
    print(f"{'='*60}\n")

    return combined


async def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run a student-tutor conversation from a benchmark entry"
    )
    parser.add_argument(
        "--entry",
        help="Path to benchmark entry JSON or JSONL file",
    )
    parser.add_argument(
        "--multi-session",
        action="store_true",
        help="Run multiple sessions for same student (requires --profile or --entries)",
    )
    parser.add_argument(
        "--profile",
        help="Profile ID to filter JSONL entries (use with --entry and --multi-session)",
    )
    parser.add_argument(
        "--entries",
        help="Comma-separated paths to entry JSON files (use with --multi-session)",
    )
    parser.add_argument(
        "--no-evolve",
        action="store_true",
        help="Disable profile evolution between sessions (multi-session only)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use auto tutor backend instead of interactive mode",
    )
    parser.add_argument(
        "--auto-backend",
        choices=["deep_tutor", "mock"],
        default="deep_tutor",
        help="Auto tutor backend (default: deep_tutor)",
    )
    parser.add_argument(
        "--deeptutor-language",
        default="en",
        help="Language for DeepTutor tools (default: en)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of student turns (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save transcript (default: benchmark/data/transcripts/)",
    )
    parser.add_argument(
        "--entry-index",
        type=int,
        default=0,
        help="When entry is a JSONL file, which entry to use (0-based, default: 0)",
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Use console input instead of editor (empty line + Enter to send)",
    )

    args = parser.parse_args()
    if args.auto:
        _suppress_noisy_auto_logs()

    if args.multi_session:
        if args.entries:
            entry_paths = [p.strip() for p in args.entries.split(",") if p.strip()]
            await run_multi_session(
                entry_paths=entry_paths,
                max_turns=args.max_turns,
                auto=args.auto,
                auto_backend=args.auto_backend,
                deeptutor_language=args.deeptutor_language,
                output_dir=args.output_dir,
                use_editor=not args.inline,
                evolve_profile=not args.no_evolve,
            )
        elif args.entry and args.profile:
            await run_multi_session(
                entry_path=args.entry,
                profile_id=args.profile,
                max_turns=args.max_turns,
                auto=args.auto,
                auto_backend=args.auto_backend,
                deeptutor_language=args.deeptutor_language,
                output_dir=args.output_dir,
                use_editor=not args.inline,
                evolve_profile=not args.no_evolve,
            )
        else:
            parser.error(
                "Multi-session requires either --entries or (--entry + --profile)"
            )
    else:
        if not args.entry:
            parser.error("Single-session mode requires --entry")
        await run_conversation(
            entry_path=args.entry,
            max_turns=args.max_turns,
            auto=args.auto,
            auto_backend=args.auto_backend,
            deeptutor_language=args.deeptutor_language,
            output_dir=args.output_dir,
            entry_index=args.entry_index,
            use_editor=not args.inline,
        )


if __name__ == "__main__":
    asyncio.run(main())
