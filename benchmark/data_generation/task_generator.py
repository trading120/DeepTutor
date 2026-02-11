#!/usr/bin/env python
"""
Task Generator - Generate learning tasks for (profile, gaps) combinations

Creates natural learning tasks that, when pursued in conversation with a tutor,
will organically expose the student's knowledge gaps.
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
    task_type_config: list[dict] | None = None,
) -> list[dict]:
    """
    Generate learning tasks for a specific (profile, gaps) combination.

    Args:
        knowledge_scope: Knowledge scope dictionary
        student_profile: Student profile dictionary
        knowledge_gaps: List of knowledge gap dictionaries for this profile
        num_tasks: Number of tasks to generate
        task_type_config: Task type configuration, e.g.:
            [{"name": "concept_understanding", "weight": 0.3}, ...]

    Returns:
        List of task dictionaries
    """
    if task_type_config is None:
        task_type_config = [
            {"name": "concept_understanding", "weight": 0.3},
            {"name": "problem_solving", "weight": 0.3},
            {"name": "application", "weight": 0.2},
            {"name": "comparison", "weight": 0.2},
        ]

    profile_id = student_profile.get("profile_id", "unknown")
    logger.info(f"Generating {num_tasks} tasks for profile '{profile_id}'")

    prompt = load_prompt("generate_tasks")

    scope_text = json.dumps(knowledge_scope, ensure_ascii=False, indent=2)
    profile_text = json.dumps(student_profile, ensure_ascii=False, indent=2)
    gaps_text = json.dumps(knowledge_gaps, ensure_ascii=False, indent=2)

    type_prefs = ", ".join(f"{t['name']}({t['weight']})" for t in task_type_config)

    user_prompt = render_prompt(
        prompt["user_template"],
        knowledge_scope=scope_text,
        student_profile=profile_text,
        knowledge_gaps=gaps_text,
        num_tasks=num_tasks,
        task_type_preferences=type_prefs,
    )

    result = await call_llm_json(
        user_prompt=user_prompt,
        system_prompt=prompt["system"],
        temperature=0.7,
        max_tokens=4096,
    )

    tasks = result.get("tasks", [])

    # Ensure task IDs
    for i, task in enumerate(tasks):
        if "task_id" not in task:
            task["task_id"] = f"task_{profile_id}_{i:02d}"

    logger.info(f"  Generated {len(tasks)} tasks: {[t.get('type', '?') for t in tasks]}")
    return tasks
