# -*- coding: utf-8 -*-
"""
Answer extraction and mathematical equivalence utilities for MATH benchmark.

Wraps the upstream `math_equivalence.is_equiv` with extra extraction helpers
so callers can go straight from raw LLM output to a boolean verdict.

Provides both regex-based extraction (fast, free) and LLM-based extraction
(more accurate, requires an LLM call) for flexible evaluation.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# Make the upstream math_equivalence importable
_MODELING_DIR = Path(__file__).resolve().parent / "math" / "modeling"
if str(_MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELING_DIR))

from math_equivalence import is_equiv  # noqa: E402

logger = logging.getLogger(__name__)


def last_boxed_only_string(string: str) -> str | None:
    """Return the last ``\\boxed{...}`` or ``\\fbox{...}`` substring (inclusive)."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    num_open = 0
    right = None
    while i < len(string):
        if string[i] == "{":
            num_open += 1
        if string[i] == "}":
            num_open -= 1
            if num_open == 0:
                right = i
                break
        i += 1

    return string[idx : right + 1] if right is not None else None


def remove_boxed(s: str | None) -> str | None:
    """Strip the ``\\boxed{`` prefix and trailing ``}``."""
    if s is None:
        return None
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    left2 = "\\fbox{"
    if s.startswith(left2) and s.endswith("}"):
        return s[len(left2) : -1]
    return s


_BOXED_RE = re.compile(r"\\boxed\{(.+?)\}", re.DOTALL)


def extract_answer(text: str) -> str | None:
    """Best-effort answer extraction from arbitrary LLM output.

    Strategy (in order):
    1. Last ``\\boxed{...}`` in the text (handles nested braces).
    2. Regex fallback for simple ``\\boxed{X}`` patterns.
    3. If the text contains ``the answer is`` / ``答案是``, grab the trailing part.
    4. Return ``None`` if nothing found.
    """
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)

    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()

    for marker in ("the answer is", "the final answer is", "答案是", "最终答案"):
        low = text.lower()
        pos = low.rfind(marker)
        if pos >= 0:
            tail = text[pos + len(marker) :].strip().rstrip(".")
            if tail:
                return tail

    return None


def check_answer(model_output: str, ground_truth: str) -> bool:
    """Return *True* if the model's extracted answer is equivalent to the ground truth."""
    predicted = extract_answer(model_output)
    if predicted is None:
        return False

    gt = extract_answer(ground_truth)
    if gt is None:
        gt = ground_truth.strip()

    try:
        return is_equiv(predicted, gt)
    except Exception:
        return predicted.strip() == gt.strip()


# ======================================================================
# LLM-based answer extraction
# ======================================================================

_LLM_EXTRACT_SYSTEM = (
    "You are a precise answer extractor. "
    "Given a math problem and a model-generated solution, "
    "extract ONLY the final answer that the model uses to answer the problem.\n\n"
    "Rules:\n"
    "- Output ONLY the answer itself — a number, expression, set, or mathematical object.\n"
    "- Do NOT include any surrounding text, labels, or prefixes "
    "(e.g. output \"5\" instead of \"n = 5\", "
    "output \"\\frac{1}{2}\" instead of \"the answer is \\frac{1}{2}\").\n"
    "- Do NOT wrap the answer in \\boxed{} or any other formatting.\n"
    "- If the solution contains multiple candidate answers, extract the one "
    "that the model presents as its final conclusion.\n"
    "- If the solution does not contain a clear answer, output: NONE"
)

_LLM_EXTRACT_USER = """\
## Math Problem
{question}

## Model's Solution
{model_output}

Extract the final answer:"""


async def extract_answer_with_llm(
    question: str,
    model_output: str,
) -> str | None:
    """Use an LLM call to extract the model's predicted answer.

    The LLM sees only the question and the model's full solution — no
    ground truth is provided.  It returns the bare final answer string.

    Falls back to regex-based ``extract_answer`` on any failure.
    """
    from src.services.llm import complete
    from src.services.llm.config import get_llm_config

    config = get_llm_config()

    user_prompt = _LLM_EXTRACT_USER.format(
        question=question,
        model_output=model_output[:6000],
    )

    try:
        response = await complete(
            prompt=user_prompt,
            system_prompt=_LLM_EXTRACT_SYSTEM,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0.0,
            max_tokens=256,
        )
        answer = response.strip()
        if not answer or answer.upper() == "NONE":
            return None
        return answer
    except Exception as exc:
        logger.warning("LLM answer extraction failed: %s", exc)
        return extract_answer(model_output)


# ======================================================================
# LLM-based equivalence judgment
# ======================================================================

_LLM_JUDGE_SYSTEM = (
    "You are a rigorous math answer grader. "
    "Given a math problem, a predicted answer, and the ground-truth answer, "
    "determine whether the predicted answer is mathematically equivalent "
    "to the ground truth.\n\n"
    "Ignore superficial differences such as:\n"
    "- Formatting: \\boxed{}, spaces, \\text{}, commas vs 'and'\n"
    "- Equivalent forms: 0.5 vs 1/2 vs \\frac{1}{2}\n"
    "- Ordering in sets or lists: {1,2,3} vs {3,2,1}\n"
    "- Notation: (-inf,0] vs (-\\infty, 0]\n\n"
    "Focus ONLY on mathematical substance.\n\n"
    "Output EXACTLY one word: CORRECT or INCORRECT"
)

_LLM_JUDGE_USER = """\
## Math Problem
{question}

## Ground-Truth Answer
{ground_truth}

## Predicted Answer
{predicted}

Verdict:"""


async def judge_answer_with_llm(
    question: str,
    predicted: str,
    ground_truth: str,
) -> bool:
    """Use an LLM to judge whether *predicted* is mathematically equivalent
    to *ground_truth*, ignoring formatting differences.

    Falls back to ``is_equiv`` on any LLM failure.
    """
    from src.services.llm import complete
    from src.services.llm.config import get_llm_config

    config = get_llm_config()

    user_prompt = _LLM_JUDGE_USER.format(
        question=question,
        ground_truth=ground_truth,
        predicted=predicted,
    )

    try:
        response = await complete(
            prompt=user_prompt,
            system_prompt=_LLM_JUDGE_SYSTEM,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0.0,
            max_tokens=16,
        )
        verdict = response.strip().upper()
        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            return True
        if "INCORRECT" in verdict:
            return False
        logger.warning("LLM judge returned ambiguous verdict: %s", verdict)
        return is_equiv(predicted, ground_truth)
    except Exception as exc:
        logger.warning("LLM judge failed, falling back to is_equiv: %s", exc)
        try:
            return is_equiv(predicted, ground_truth)
        except Exception:
            return predicted.strip() == ground_truth.strip()
