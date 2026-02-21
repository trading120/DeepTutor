# -*- coding: utf-8 -*-
"""
MATH Benchmark Runner — evaluate DeepTutor's LLM on the MATH dataset.

Dataset: HuggingFace ``qwedsacf/competition_math`` (12 500 problems).
First run downloads the parquet file and caches it locally under
``benchmark/iso_solve/data/``.

Usage examples (run from project root with the hkuds conda env):

  # Dry-run — verify data loading, no LLM calls:
  python -m benchmark.iso_solve.run_benchmark --dry-run --limit 5

  # Direct LLM mode (fast), 50 random samples, all subjects:
  python -m benchmark.iso_solve.run_benchmark --mode direct --limit 50

  # Full pipeline mode (Plan→ReAct→Write), code_execute only, 20 samples:
  python -m benchmark.iso_solve.run_benchmark --mode pipeline \
      --tools code_execute --limit 20

  # Pipeline mode with all tools, algebra only, Level 1-3:
  python -m benchmark.iso_solve.run_benchmark --mode pipeline \
      --subjects Algebra --levels 1 2 3 --limit 20

  # Use a pre-downloaded parquet or local JSON files:
  python -m benchmark.iso_solve.run_benchmark --mode direct \
      --dataroot benchmark/iso_solve/data/competition_math.parquet --limit 50

  # With config file:
  python -m benchmark.iso_solve.run_benchmark --mode pipeline \
      --config benchmark/iso_solve/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / "DeepTutor.env", override=False)
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from benchmark.iso_solve.answer_utils import (
    check_answer,
    extract_answer,
    extract_answer_with_llm,
    is_equiv,
    judge_answer_with_llm,
    last_boxed_only_string,
    remove_boxed,
)

logger = logging.getLogger("math_benchmark")

# ======================================================================
# Data structures
# ======================================================================

SUBJECTS = [
    "Prealgebra",
    "Algebra",
    "Number Theory",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Precalculus",
]


@dataclass
class MathProblem:
    problem: str
    solution: str
    level: int | None
    subject: str
    source_file: str = ""

    @property
    def ground_truth(self) -> str | None:
        """Extract the boxed answer from the reference solution."""
        boxed = last_boxed_only_string(self.solution)
        return remove_boxed(boxed) if boxed else None


@dataclass
class EvalResult:
    problem: MathProblem
    model_output: str
    predicted_answer: str | None
    ground_truth: str | None
    correct: bool
    elapsed_sec: float
    error: str | None = None


@dataclass
class BenchmarkReport:
    mode: str
    model: str
    timestamp: str
    total: int = 0
    correct: int = 0
    skipped: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0
    results: list[EvalResult] = field(default_factory=list)
    by_subject: dict[str, dict] = field(default_factory=dict)
    by_level: dict[int, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def add(self, r: EvalResult) -> None:
        self.results.append(r)
        self.total += 1
        if r.error:
            self.errors += 1
            return
        if r.correct:
            self.correct += 1

        subj = r.problem.subject
        if subj not in self.by_subject:
            self.by_subject[subj] = {"total": 0, "correct": 0}
        self.by_subject[subj]["total"] += 1
        if r.correct:
            self.by_subject[subj]["correct"] += 1

        lvl = r.problem.level
        if lvl is not None:
            if lvl not in self.by_level:
                self.by_level[lvl] = {"total": 0, "correct": 0}
            self.by_level[lvl]["total"] += 1
            if r.correct:
                self.by_level[lvl]["correct"] += 1

    def summary_lines(self) -> list[str]:
        lines = [
            "=" * 60,
            f"MATH Benchmark Report — {self.mode} mode",
            f"Model: {self.model}",
            f"Time:  {self.timestamp}",
            "=" * 60,
            f"Overall: {self.correct}/{self.total} = {self.accuracy:.3f}  "
            f"(errors={self.errors}, skipped={self.skipped})",
            f"Wall time: {self.elapsed_sec:.1f}s",
            "",
            "--- By Subject ---",
        ]
        for subj in SUBJECTS:
            if subj in self.by_subject:
                s = self.by_subject[subj]
                acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
                lines.append(f"  {subj:30s} {s['correct']:3d}/{s['total']:3d} = {acc:.3f}")
        lines.append("")
        lines.append("--- By Level ---")
        for lvl in sorted(self.by_level):
            s = self.by_level[lvl]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            lines.append(f"  Level {lvl}  {s['correct']:3d}/{s['total']:3d} = {acc:.3f}")
        lines.append("=" * 60)
        return lines

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "model": self.model,
            "timestamp": self.timestamp,
            "overall": {
                "total": self.total,
                "correct": self.correct,
                "accuracy": round(self.accuracy, 4),
                "errors": self.errors,
                "skipped": self.skipped,
                "elapsed_sec": round(self.elapsed_sec, 2),
            },
            "by_subject": {
                k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0}
                for k, v in self.by_subject.items()
            },
            "by_level": {
                str(k): {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0}
                for k, v in sorted(self.by_level.items())
            },
            "results": [
                {
                    "problem": r.problem.problem[:200],
                    "subject": r.problem.subject,
                    "level": r.problem.level,
                    "predicted": r.predicted_answer,
                    "ground_truth": r.ground_truth,
                    "correct": r.correct,
                    "elapsed_sec": round(r.elapsed_sec, 2),
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ======================================================================
# Data loading
# ======================================================================

_HF_PARQUET_URL = (
    "hf://datasets/qwedsacf/competition_math/data/"
    "train-00000-of-00001-7320a6f3aba8ebd2.parquet"
)
_LOCAL_CACHE_DIR = Path(__file__).resolve().parent / "data"
_LOCAL_CACHE_FILE = _LOCAL_CACHE_DIR / "competition_math.parquet"


def _parse_level(level_str: Any) -> int | None:
    if isinstance(level_str, int):
        return level_str
    if isinstance(level_str, str) and "Level " in level_str:
        try:
            return int(level_str.split("Level ")[1])
        except (ValueError, IndexError):
            pass
    return None


def _df_to_problems(df: "pd.DataFrame", source_tag: str = "") -> list[MathProblem]:
    """Convert a pandas DataFrame to a list of MathProblem."""
    problems: list[MathProblem] = []
    for idx, row in df.iterrows():
        level = _parse_level(row.get("level", ""))
        problems.append(MathProblem(
            problem=row["problem"],
            solution=row["solution"],
            level=level,
            subject=row.get("type", "Unknown"),
            source_file=f"{source_tag}#{idx}",
        ))
    return problems


def load_math_from_local(dataroot: str) -> list[MathProblem]:
    """Load MATH problems from local JSON files (glob pattern or directory)."""
    import glob as globmod

    if "*" in dataroot:
        files = globmod.glob(dataroot)
    else:
        root = Path(dataroot)
        files = sorted(str(f) for f in root.rglob("*.json"))

    problems: list[MathProblem] = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            problems.append(MathProblem(
                problem=data["problem"],
                solution=data["solution"],
                level=_parse_level(data.get("level", "")),
                subject=data.get("type", "Unknown"),
                source_file=fpath,
            ))
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")
    return problems


def load_math_from_parquet(
    url: str = _HF_PARQUET_URL,
    cache: bool = True,
) -> list[MathProblem]:
    """Load MATH dataset from a HuggingFace parquet URL (with local caching).

    First run downloads from HF and caches to ``benchmark/iso_solve/data/``.
    Subsequent runs load directly from the local cache.
    """
    import pandas as pd

    if cache and _LOCAL_CACHE_FILE.exists():
        logger.info(f"Loading cached dataset from {_LOCAL_CACHE_FILE}")
        df = pd.read_parquet(_LOCAL_CACHE_FILE)
    else:
        logger.info(f"Downloading MATH dataset from {url} ...")
        df = pd.read_parquet(url)
        if cache:
            _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(_LOCAL_CACHE_FILE, index=False)
            logger.info(f"Cached {len(df)} problems to {_LOCAL_CACHE_FILE}")

    # Filter out invalid rows (e.g. "Level ?")
    df = df[df["level"] != "Level ?"].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} problems (columns: {df.columns.tolist()})")
    return _df_to_problems(df, source_tag="competition_math")


def filter_problems(
    problems: list[MathProblem],
    subjects: list[str] | None = None,
    levels: list[int] | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[MathProblem]:
    """Filter and optionally sample a subset of problems."""
    filtered = problems
    if subjects:
        subj_set = set(subjects)
        filtered = [p for p in filtered if p.subject in subj_set]
    if levels:
        lvl_set = set(levels)
        filtered = [p for p in filtered if p.level in lvl_set]
    if limit and limit < len(filtered):
        rng = random.Random(seed)
        filtered = rng.sample(filtered, limit)
    return filtered


# ======================================================================
# Evaluation — Direct LLM mode
# ======================================================================

DIRECT_SYSTEM_PROMPT = (
    "You are a precise mathematics problem solver. "
    "Solve the given problem step by step, then put your final answer "
    "inside \\boxed{}. For example: \\boxed{42}."
)


async def _judge(
    question: str,
    predicted: str | None,
    ground_truth: str | None,
    use_llm: bool,
) -> bool:
    """Compare predicted vs ground_truth.

    When *use_llm* is True, delegates to ``judge_answer_with_llm`` which
    understands mathematical equivalence beyond surface-level string matching.
    Otherwise falls back to the classic ``is_equiv`` normalised comparison.
    """
    if predicted is None or ground_truth is None:
        return False

    if use_llm:
        return await judge_answer_with_llm(question, predicted, ground_truth)

    try:
        return is_equiv(predicted, ground_truth)
    except Exception:
        return predicted.strip() == ground_truth.strip()


async def eval_direct(
    problem: MathProblem,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    use_llm_extract: bool = False,
    use_llm_judge: bool = False,
) -> EvalResult:
    """Evaluate a single problem using direct LLM call."""
    from src.services.llm import complete
    from src.services.llm.config import get_llm_config

    config = get_llm_config()
    gt = problem.ground_truth

    t0 = time.time()
    try:
        response = await complete(
            prompt=problem.problem,
            system_prompt=DIRECT_SYSTEM_PROMPT,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0

        predicted = extract_answer(response)
        if use_llm_extract:
            predicted = await extract_answer_with_llm(
                problem.problem, response,
            )

        correct = await _judge(
            problem.problem, predicted, gt, use_llm_judge,
        )

        return EvalResult(
            problem=problem,
            model_output=response,
            predicted_answer=predicted,
            ground_truth=gt,
            correct=correct,
            elapsed_sec=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        return EvalResult(
            problem=problem,
            model_output="",
            predicted_answer=None,
            ground_truth=gt,
            correct=False,
            elapsed_sec=elapsed,
            error=str(e),
        )


# ======================================================================
# Evaluation — Full Pipeline mode
# ======================================================================

def _build_pipeline_registry(
    tools: list[str] | None,
    language: str,
) -> "ToolRegistry":
    """Build a ToolRegistry for benchmark evaluation.

    When *tools* is None, all default tools are loaded.
    When *tools* is an explicit list (e.g. ["code_execute"]),
    only those tools (plus the mandatory done/replan controls) are kept.
    """
    from src.agents.solve.tools import ToolRegistry

    if tools is None:
        return ToolRegistry.create_default(language=language)
    return ToolRegistry.create_from_names(tools, language=language)


async def eval_pipeline(
    problem: MathProblem,
    workspace: str,
    language: str = "en",
    tools: list[str] | None = None,
    use_llm_extract: bool = False,
    use_llm_judge: bool = False,
) -> EvalResult:
    """Evaluate a single problem using the Plan → ReAct → Write pipeline.

    Args:
        problem: The math problem to solve.
        workspace: Directory for pipeline output artefacts.
        language: Prompt language (``"en"`` or ``"zh"``).
        tools: Explicit tool list (e.g. ``["code_execute"]``).
               ``None`` loads all default tools.
               ``done`` and ``replan`` are always available.
        use_llm_extract: Use LLM to extract the predicted answer.
        use_llm_judge: Use LLM to judge mathematical equivalence.
    """
    from src.agents.solve import MainSolver

    registry = _build_pipeline_registry(tools, language)
    has_rag = registry.get("rag_search") is not None

    gt = problem.ground_truth
    t0 = time.time()
    try:
        solver = MainSolver(
            kb_name="" if not has_rag else "__benchmark__",
            language=language,
            output_base_dir=workspace,
            tool_registry=registry,
            disable_memory=True,
        )
        await solver.ainit()
        result = await solver.solve(problem.problem)
        elapsed = time.time() - t0

        final_answer = result.get("final_answer", "")

        predicted = extract_answer(final_answer)
        if use_llm_extract:
            predicted = await extract_answer_with_llm(
                problem.problem, final_answer,
            )

        correct = await _judge(
            problem.problem, predicted, gt, use_llm_judge,
        )

        return EvalResult(
            problem=problem,
            model_output=final_answer,
            predicted_answer=predicted,
            ground_truth=gt,
            correct=correct,
            elapsed_sec=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        return EvalResult(
            problem=problem,
            model_output="",
            predicted_answer=None,
            ground_truth=gt,
            correct=False,
            elapsed_sec=elapsed,
            error=str(e),
        )


# ======================================================================
# Main runner
# ======================================================================

async def run_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    """Run the full benchmark and return a report."""
    # --- Load config ---
    bench_cfg: dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            bench_cfg = yaml.safe_load(f) or {}

    # --- Load data ---
    if args.dataroot:
        dataroot = args.dataroot
        if dataroot.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(dataroot)
            problems = _df_to_problems(df, source_tag=dataroot)
        else:
            problems = load_math_from_local(dataroot)
    else:
        parquet_url = bench_cfg.get("dataset", {}).get("parquet_url", _HF_PARQUET_URL)
        problems = load_math_from_parquet(url=parquet_url)

    subjects = args.subjects or bench_cfg.get("filter", {}).get("subjects")
    levels = args.levels or bench_cfg.get("filter", {}).get("levels")
    limit = args.limit or bench_cfg.get("filter", {}).get("limit")
    seed = args.seed if hasattr(args, "seed") else bench_cfg.get("filter", {}).get("seed", 42)

    problems = filter_problems(problems, subjects=subjects, levels=levels, limit=limit, seed=seed)
    logger.info(f"Evaluating {len(problems)} problems (mode={args.mode})")

    if not problems:
        logger.error("No problems to evaluate after filtering.")
        sys.exit(1)

    # --- Determine model name ---
    model_name = "unknown"
    try:
        from src.services.llm.config import get_llm_config
        model_name = get_llm_config().model or "unknown"
    except Exception:
        model_name = os.getenv("LLM_MODEL", "unknown")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = BenchmarkReport(mode=args.mode, model=model_name, timestamp=timestamp)

    # --- Output directory ---
    output_dir = Path(args.output or bench_cfg.get("output_dir", "benchmark/iso_solve/results"))
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dry run ---
    if args.dry_run:
        logger.info("=== DRY RUN — no LLM calls ===")
        for i, p in enumerate(problems[:5]):
            gt = p.ground_truth
            logger.info(f"  [{i+1}] level={p.level} subject={p.subject} gt={gt}")
            logger.info(f"       {p.problem[:120]}...")
        logger.info(f"Total problems after filter: {len(problems)}")
        return report

    # --- Eval loop ---
    temperature = bench_cfg.get("llm", {}).get("temperature", 0.0)
    max_tokens = bench_cfg.get("llm", {}).get("max_tokens", 4096)
    concurrency = bench_cfg.get("concurrency", 1)
    use_llm_extract: bool = (
        args.llm_extract
        if args.llm_extract is not None
        else bench_cfg.get("llm_extract", False)
    )
    use_llm_judge: bool = (
        args.llm_judge
        if args.llm_judge is not None
        else bench_cfg.get("llm_judge", False)
    )

    if use_llm_extract:
        logger.info("LLM answer extraction: ENABLED")
    if use_llm_judge:
        logger.info("LLM answer judging:    ENABLED")

    pipeline_workspace = str(output_dir / "pipeline_workspaces")
    t_start = time.time()

    if args.mode == "direct":
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(idx: int, prob: MathProblem) -> EvalResult:
            async with sem:
                logger.info(f"[{idx+1}/{len(problems)}] Solving: {prob.problem[:80]}...")
                r = await eval_direct(
                    prob,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_llm_extract=use_llm_extract,
                    use_llm_judge=use_llm_judge,
                )
                status = "✓" if r.correct else ("✗" if not r.error else "ERR")
                logger.info(
                    f"  {status} predicted={r.predicted_answer}  gt={r.ground_truth}  "
                    f"({r.elapsed_sec:.1f}s)"
                )
                return r

        tasks = [_run_one(i, p) for i, p in enumerate(problems)]
        results = await asyncio.gather(*tasks)
        for r in results:
            report.add(r)

    elif args.mode == "pipeline":
        pipeline_cfg = bench_cfg.get("pipeline", {})
        pipeline_tools: list[str] | None = (
            args.tools
            or pipeline_cfg.get("tools")
        )
        pipeline_lang = pipeline_cfg.get("language", "en")
        pipeline_concurrency = pipeline_cfg.get("concurrency", 1)

        if pipeline_tools is not None:
            logger.info(f"Pipeline tools: {pipeline_tools} (+ done, replan)")
        else:
            logger.info("Pipeline tools: all defaults")
        logger.info(f"Pipeline concurrency: {pipeline_concurrency}")

        sem = asyncio.Semaphore(pipeline_concurrency)

        async def _run_pipeline_one(idx: int, prob: MathProblem) -> EvalResult:
            async with sem:
                logger.info(f"[{idx+1}/{len(problems)}] Solving (pipeline): {prob.problem[:80]}...")
                ws = os.path.join(pipeline_workspace, f"prob_{idx:04d}")
                os.makedirs(ws, exist_ok=True)
                r = await eval_pipeline(
                    prob,
                    workspace=ws,
                    language=pipeline_lang,
                    tools=pipeline_tools,
                    use_llm_extract=use_llm_extract,
                    use_llm_judge=use_llm_judge,
                )
                status = "✓" if r.correct else ("✗" if not r.error else "ERR")
                logger.info(
                    f"  {status} predicted={r.predicted_answer}  gt={r.ground_truth}  "
                    f"({r.elapsed_sec:.1f}s)"
                )
                return r

        tasks = [_run_pipeline_one(i, p) for i, p in enumerate(problems)]
        results = await asyncio.gather(*tasks)
        for r in results:
            report.add(r)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    report.elapsed_sec = time.time() - t_start

    # --- Print & save report ---
    for line in report.summary_lines():
        print(line)

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {report_path}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report.summary_lines()))

    return report


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MATH Benchmark — evaluate LLM on the MATH dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["direct", "pipeline"], default="direct",
        help="Evaluation mode: 'direct' (raw LLM) or 'pipeline' (Plan→ReAct→Write)",
    )
    parser.add_argument(
        "--dataroot", type=str, default=None,
        help="Local path or glob to MATH JSON files. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help=f"Filter by subject(s). Choices: {SUBJECTS}",
    )
    parser.add_argument("--levels", nargs="+", type=int, default=None, help="Filter by level(s) 1-5")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to benchmark config YAML")
    parser.add_argument(
        "--tools", nargs="+", default=None,
        help="Pipeline mode: tools to enable (e.g. code_execute). "
             "done/replan are always available. Omit to use all defaults.",
    )
    parser.add_argument(
        "--llm-extract", action="store_true", default=None,
        help="Use LLM to extract predicted answers (more accurate than regex).",
    )
    parser.add_argument(
        "--llm-judge", action="store_true", default=None,
        help="Use LLM to judge mathematical equivalence (more accurate than string matching).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data only, no LLM calls")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
