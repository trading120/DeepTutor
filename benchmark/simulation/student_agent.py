#!/usr/bin/env python
"""
Student Agent - LLM-based student role-playing for tutor evaluation

Takes a benchmark entry (profile + gaps + task) and role-plays as the student
in a multi-turn conversation with a tutor.

The agent:
- Initiates conversation with task.initial_message
- Responds to tutor messages while staying in character
- Naturally exhibits knowledge gaps (misconceptions, incomplete understanding)
- Gradually learns from good explanations but doesn't accept corrections instantly

Usage:
    agent = StudentAgent.from_entry(entry)
    first_msg = agent.initial_message()
    response, task_complete = await agent.respond(tutor_reply)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("benchmark.student_agent")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Marker the student outputs when the learning task is complete (end of response)
# Match flexibly: [TASK_COMPLETE], [ TASK_COMPLETE ], etc.
TASK_COMPLETE_MARKER = "[TASK_COMPLETE]"
TASK_COMPLETE_RE = re.compile(r"\[\s*TASK_COMPLETE\s*\]", re.IGNORECASE)


def _load_student_prompt_template() -> str:
    """Load the student system prompt template."""
    path = PROJECT_ROOT / "benchmark" / "prompts" / "student_system.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["system"]


def _format_list(items: list[str], bullet: str = "-") -> str:
    """Format a list of strings as bullet points."""
    if not items:
        return "(none)"
    return "\n".join(f"  {bullet} {item}" for item in items)


def _build_beliefs_text(gaps: list[dict]) -> str:
    """
    Convert knowledge gaps into first-person belief statements.

    This is the key trick: gaps are encoded as what the student *believes*,
    not as what's wrong. This prevents the LLM from being self-aware about errors.
    """
    if not gaps:
        return "(No specific beliefs to note.)"

    beliefs = []
    for gap in gaps:
        gap_type = gap.get("type", "unknown")
        concept = gap.get("target_concept", "")
        description = gap.get("description", "")
        manifestation = gap.get("manifestation", "")

        if gap_type == "misconception":
            beliefs.append(
                f"You believe the following about {concept}: {description}\n"
                f"  This feels obviously correct to you. "
                f"In conversation, you would naturally: {manifestation}"
            )
        elif gap_type == "incomplete":
            beliefs.append(
                f"Regarding {concept}: {description}\n"
                f"  You feel you understand this topic reasonably well, "
                f"but haven't thought deeply about edge cases. "
                f"In conversation, you might: {manifestation}"
            )
        elif gap_type == "missing":
            beliefs.append(
                f"You have never encountered {concept}. {description}\n"
                f"  If this comes up, you would: {manifestation}"
            )
        else:
            beliefs.append(f"About {concept}: {description}")

    return "\n\n".join(f"[Belief {i+1}]\n{b}" for i, b in enumerate(beliefs))


class StudentAgent:
    """
    An LLM-powered student agent for tutor evaluation.

    Maintains conversation history and generates in-character student responses.
    """

    def __init__(
        self,
        system_prompt: str,
        first_message: str,
        entry_id: str = "unknown",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.system_prompt = system_prompt
        self.first_message = first_message
        self.entry_id = entry_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Conversation history: list of {"role": "user"/"assistant", "content": "..."}
        # From the tutor's perspective: user=student, assistant=tutor
        # But from the student LLM's perspective: user=tutor, assistant=student
        # We store it from the STUDENT's perspective (student=assistant, tutor=user)
        self.history: list[dict[str, str]] = []
        self.turn_count: int = 0

    @classmethod
    def from_entry(cls, entry: dict, **kwargs) -> "StudentAgent":
        """
        Create a StudentAgent from a benchmark entry.

        Args:
            entry: Benchmark entry dict with profile, gaps, task
            **kwargs: Override temperature, max_tokens, etc.

        Returns:
            Configured StudentAgent
        """
        profile = entry.get("profile", {})
        gaps = entry.get("gaps", [])
        task = entry.get("task", {})
        entry_id = entry.get("entry_id", "unknown")

        # Build system prompt from template
        template = _load_student_prompt_template()

        knowledge_state = profile.get("knowledge_state", {})

        system_prompt = template.format(
            personality=profile.get("personality", "A typical student."),
            education_background=profile.get(
                "education_background", "An undergraduate student."
            ),
            learning_purpose=profile.get(
                "learning_purpose", "To learn the material for an upcoming exam."
            ),
            known_well=_format_list(knowledge_state.get("known_well", [])),
            partially_known=_format_list(knowledge_state.get("partially_known", [])),
            unknown=_format_list(knowledge_state.get("unknown", [])),
            beliefs=_build_beliefs_text(gaps),
        )

        first_message = task.get("initial_message", "Hi, I need help with this topic.")

        return cls(
            system_prompt=system_prompt,
            first_message=first_message,
            entry_id=entry_id,
            **kwargs,
        )

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "StudentAgent":
        """
        Create a StudentAgent from a benchmark entry JSON file.

        Args:
            path: Path to the entry JSON file
            **kwargs: Override temperature, max_tokens, etc.

        Returns:
            Configured StudentAgent
        """
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        return cls.from_entry(entry, **kwargs)

    def initial_message(self) -> str:
        """
        Get the student's first message (from task.initial_message).

        Also records it in history.
        """
        self.history.append({"role": "assistant", "content": self.first_message})
        self.turn_count = 1
        return self.first_message

    async def respond(self, tutor_message: str) -> tuple[str, bool]:
        """
        Generate a student response to a tutor message.

        Args:
            tutor_message: The tutor's latest message

        Returns:
            Tuple of (response_text, task_complete). task_complete is True when
            the student outputs [TASK_COMPLETE] indicating the learning task is done.
        """
        # Record tutor message
        self.history.append({"role": "user", "content": tutor_message})

        # Build messages array for LLM call
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)

        # Call LLM
        from src.services.llm import factory

        response = await factory.complete(
            prompt="",  # Not used when messages is provided
            system_prompt="",  # Not used when messages is provided
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check for task-complete marker and strip it
        task_complete = bool(TASK_COMPLETE_RE.search(response))
        cleaned = TASK_COMPLETE_RE.sub("", response).strip()

        # Record cleaned response in history (no marker in transcript)
        self.history.append({"role": "assistant", "content": cleaned})
        self.turn_count += 1

        return (cleaned, task_complete)

    def get_transcript(self) -> list[dict[str, str]]:
        """
        Get the full conversation transcript.

        Returns:
            List of messages with roles normalized to "student"/"tutor".
        """
        transcript = []
        for msg in self.history:
            role = "student" if msg["role"] == "assistant" else "tutor"
            transcript.append({"role": role, "content": msg["content"]})
        return transcript

    def get_metadata(self) -> dict[str, Any]:
        """Get agent metadata for saving alongside transcript."""
        return {
            "entry_id": self.entry_id,
            "turn_count": self.turn_count,
            "system_prompt_length": len(self.system_prompt),
        }
