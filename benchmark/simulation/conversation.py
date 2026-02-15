#!/usr/bin/env python
"""
Conversation Runner - Run multi-turn student-tutor conversations

Supports two modes:
  1. Interactive: Human plays the tutor, types responses in terminal
  2. Auto: Another LLM plays the tutor (simple mock for testing)

Usage:
  # Interactive mode (editor by default; use --inline for console input):
  python -m benchmark.simulation.conversation --entry path/to/entry.json

  # Console input (empty line + Enter to send; paste may truncate long content):
  python -m benchmark.simulation.conversation --entry path/to/entry.json --inline

  # Auto mode (LLM plays the tutor; use --max-turns 5 if you hit timeout):
  python -m benchmark.simulation.conversation --entry path/to/entry.json --auto
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

from benchmark.simulation.student_agent import StudentAgent

try:
    from src.services.llm.exceptions import LLMAPIError
except ImportError:
    LLMAPIError = Exception  # fallback if not in path

# Delimiter to separate student context from tutor response in editor
_EDITOR_DELIMITER = "\n\n--- Type your response below this line ---\n\n"

logger = logging.getLogger("benchmark.conversation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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


# Simple tutor system prompt for auto mode
MOCK_TUTOR_SYSTEM = (
    "You are a helpful, patient, and knowledgeable tutor. "
    "A student is asking you for help. Respond clearly and helpfully. "
    "Ask follow-up questions to check understanding. "
    "If the student shows a misconception, gently guide them toward the correct understanding "
    "rather than just stating the answer. Use examples when helpful. "
    "Keep responses focused and not too long (2-5 sentences typically)."
)


async def mock_tutor_respond(
    student_message: str,
    history: list[dict[str, str]],
) -> str:
    """
    Generate a mock tutor response using LLM.

    Args:
        student_message: The student's latest message
        history: Conversation history from tutor's perspective
            (role="user" for student, role="assistant" for tutor)

    Returns:
        Tutor response string
    """
    from src.services.llm import factory

    # Build messages from tutor's perspective
    messages = [{"role": "system", "content": MOCK_TUTOR_SYSTEM}]
    messages.extend(history)
    messages.append({"role": "user", "content": student_message})

    response = await factory.complete(
        prompt="",
        system_prompt="",
        messages=messages,
        temperature=0.5,
        max_tokens=1024,
    )

    return response


async def run_conversation(
    entry_path: str | Path,
    max_turns: int = 20,
    auto: bool = False,
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

    print(f"\n{'='*60}")
    print(f"Conversation: {entry_id}")
    print(f"Mode: {'auto (LLM tutor)' if auto else 'interactive (you are the tutor)'}")
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
                tutor_msg = await mock_tutor_respond(student_msg, tutor_history)
            except (LLMAPIError, asyncio.TimeoutError, TimeoutError) as e:
                logger.error("Mock tutor LLM failed (timeout or API error): %s", e)
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

        # Get student response
        student_msg, task_complete = await agent.respond(tutor_msg)
        print(f"[Student] {student_msg}\n")
        if task_complete:
            print("[System] Student indicated task complete. Ending conversation.")
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
        required=True,
        help="Path to benchmark entry JSON file",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use LLM mock tutor instead of interactive mode",
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

    await run_conversation(
        entry_path=args.entry,
        max_turns=args.max_turns,
        auto=args.auto,
        output_dir=args.output_dir,
        entry_index=args.entry_index,
        use_editor=not args.inline,
    )


if __name__ == "__main__":
    asyncio.run(main())
