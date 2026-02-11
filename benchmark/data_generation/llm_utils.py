#!/usr/bin/env python
"""
LLM Utilities for Benchmark Data Generation

Provides:
- Prompt template loading (from benchmark/prompts/*.yaml)
- Unified LLM calling with JSON parsing (via src.services.llm.factory)
- Template rendering
"""

import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger("benchmark.llm_utils")

# Project root: benchmark/data_generation/llm_utils.py → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_prompt(prompt_name: str) -> dict:
    """
    Load a prompt template from benchmark/prompts/.

    Args:
        prompt_name: Prompt file name without extension (e.g., "generate_profile")

    Returns:
        dict with 'system' and 'user_template' keys
    """
    prompt_path = PROJECT_ROOT / "benchmark" / "prompts" / f"{prompt_name}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    with open(prompt_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


async def call_llm(
    user_prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> str:
    """
    Call LLM via the project's unified LLM factory.

    Uses src.services.llm.factory.complete() which handles provider routing,
    retry with exponential backoff, etc.

    Args:
        user_prompt: User prompt text
        system_prompt: System prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments (model, api_key, base_url overrides)

    Returns:
        LLM response text
    """
    from src.services.llm import factory

    return await factory.complete(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


async def call_llm_json(
    user_prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> dict:
    """
    Call LLM and parse response as JSON.

    Attempts multiple strategies to extract valid JSON from the response.

    Args:
        user_prompt: User prompt text
        system_prompt: System prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        **kwargs: Additional arguments passed to call_llm

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If response cannot be parsed as JSON
    """
    response = await call_llm(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return extract_json(response)


def extract_json(text: str) -> dict:
    """
    Extract JSON from text that may contain markdown code fences or extra text.

    Args:
        text: Raw text potentially containing JSON

    Returns:
        Parsed JSON dictionary
    """
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code fences
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost JSON object
    # Find the first '{' and the last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Cannot extract JSON from LLM response", text, 0)


def render_prompt(template: str, **kwargs) -> str:
    """
    Render a prompt template with the given variables.

    Args:
        template: Template string with {variable} placeholders
        **kwargs: Variables to substitute

    Returns:
        Rendered prompt string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")
