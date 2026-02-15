#!/usr/bin/env python
"""
Task Generator - Generate learning tasks for a profile's knowledge gaps

Creates natural learning tasks that, when pursued in conversation with a tutor,
will organically expose one or more of the student's knowledge gaps.
Each task may target one or multiple gaps — the LLM decides.
"""

import json
import logging

from benchmark.data_generation.llm_utils import call_llm_json, load_prompt, render_prompt

logger = logging.getLogger("benchmark.task_generator")


async def generate_tasks(
    knowledge_scope: dict,
    student_profile: dict,
    knowledge_gaps: list[dict],
    num_tasks: int = 2,
) -> list[dict]:
    """
    Generate learning tasks for a (profile, gaps) combination.

    The LLM receives all gaps and decides which gaps each task targets.
    Each task specifies its target_gaps (list of gap_ids).

    Args:
        knowledge_scope: Knowledge scope dictionary
        student_profile: Student profile dictionary
        knowledge_gaps: All knowledge gap dictionaries for this profile
        num_tasks: Number of tasks to generate

    Returns:
        List of task dictionaries, each with a 'target_gaps' field
    """
    profile_id = student_profile.get("profile_id", "unknown")
    logger.info(f"Generating {num_tasks} tasks for profile '{profile_id}' ({len(knowledge_gaps)} gaps)")

    prompt = load_prompt("generate_tasks")

    scope_text = json.dumps(knowledge_scope, ensure_ascii=False, indent=2)
    profile_text = json.dumps(student_profile, ensure_ascii=False, indent=2)
    gaps_text = json.dumps(knowledge_gaps, ensure_ascii=False, indent=2)

    user_prompt = render_prompt(
        prompt["user_template"],
        knowledge_scope=scope_text,
        student_profile=profile_text,
        knowledge_gaps=gaps_text,
        num_tasks=num_tasks,
    )

    result = await call_llm_json(
        user_prompt=user_prompt,
        system_prompt=prompt["system"],
        temperature=0.7,
        max_tokens=4096,
    )

    tasks = result.get("tasks", [])

    # If the LLM returned a single task object instead of a list, wrap it
    if not isinstance(tasks, list):
        tasks = [result] if "task_id" in result or "title" in result else []

    # Ensure task IDs and validate target_gaps
    gap_ids = {g.get("gap_id") for g in knowledge_gaps}
    for i, task in enumerate(tasks):
        if "task_id" not in task:
            task["task_id"] = f"task_{profile_id}_{i:02d}"

        # Ensure target_gaps is a list of valid gap_ids
        target = task.get("target_gaps", [])
        if isinstance(target, str):
            target = [target]
        task["target_gaps"] = [gid for gid in target if gid in gap_ids]

        if not task["target_gaps"]:
            logger.warning(f"  Task {task['task_id']} has no valid target_gaps")

    logger.info(f"  Generated {len(tasks)} tasks: {[t.get('type', '?') for t in tasks]}")
    return tasks
