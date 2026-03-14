#!/usr/bin/env python3
"""
Build evaluation summaries from step3 outputs.

Outputs:
1) Per-KB summary:
   <output_root>/evaluations/<kb_name>/summary.json
2) Global manifest summary:
   <output_root>/manifests/eval_summary_manifest.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "data" / "bench_pipeline"


def _parse_names(raw: str) -> list[str]:
    return sorted(set(n.strip() for n in raw.split(",") if n.strip()))


def _safe_avg(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def _fmt_num(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    return "-"


def _fmt_ratio_pct(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{100.0 * float(v):.2f}%"
    return "-"


def _build_backend_markdown_table(by_backend: dict[str, dict[str, Any]]) -> str:
    headers = [
        "Backend",
        "Profiles",
        "Faith",
        "Personal.",
        "Applic.",
        "Vivid.",
        "Logic",
        "Faith Fail",
        "Teach Fail",
        "PQ Fail",
        "PQ Ground",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for backend in sorted(by_backend.keys()):
        b = by_backend[backend]
        pq = b.get("practice_questions", {}) or {}
        row = [
            backend,
            _fmt_num(b.get("num_profiles")),
            _fmt_num(b.get("avg_faithfulness")),
            _fmt_num(b.get("avg_personalization")),
            _fmt_num(b.get("avg_applicability")),
            _fmt_num(b.get("avg_vividness")),
            _fmt_num(b.get("avg_logical_depth")),
            _fmt_ratio_pct(b.get("faithfulness_eval_failed_ratio")),
            _fmt_ratio_pct(b.get("teaching_quality_eval_failed_ratio")),
            _fmt_ratio_pct(pq.get("eval_failed_ratio")),
            _fmt_num(pq.get("avg_groundedness")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _build_markdown_report(
    *,
    title: str,
    output_root: Path,
    num_eval_files: int,
    by_backend: dict[str, dict[str, Any]],
) -> str:
    table = _build_backend_markdown_table(by_backend)
    return (
        f"# {title}\n\n"
        f"- Generated at: {datetime.now().isoformat()}\n"
        f"- Output root: `{output_root}`\n"
        f"- Eval files: {num_eval_files}\n"
        f"- Backends: {len(by_backend)}\n\n"
        "## Metrics Table\n\n"
        f"{table}\n"
    )


def _extract_eval_summary(eval_data: dict) -> dict:
    # Multi-session evals store aggregated metrics under "aggregate".
    if isinstance(eval_data.get("aggregate"), dict):
        agg = eval_data.get("aggregate", {})
        summary = {
            "turn_count": agg.get("turn_count", {}),
            "source_faithfulness": agg.get("source_faithfulness", {}),
            "teaching_quality": agg.get("teaching_quality", {}),
        }
        pq = agg.get("practice_questions")
        if pq:
            summary["practice_questions"] = pq
        return summary

    # Single-session evals store metrics directly under "metrics".
    metrics = eval_data.get("metrics", {}) if isinstance(eval_data, dict) else {}
    summary = {
        "turn_count": metrics.get("turn_count", {}),
        "source_faithfulness": metrics.get("source_faithfulness", {}),
        "teaching_quality": metrics.get("teaching_quality", {}),
    }
    pq = metrics.get("practice_questions")
    if pq:
        # In single-session output, practice_questions has {"summary": ...}.
        summary["practice_questions"] = pq.get("summary", pq)
    return summary


def _build_backend_aggregate(eval_file_records: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for rec in eval_file_records:
        backend = rec["backend"]
        grouped.setdefault(
            backend,
            {
                "num_eval_files": 0,
                "num_profiles": 0,
                "profiles": set(),
                "paired_turns_total": 0,
                "tutor_turns_total": 0,
                "faith_total_turns": 0,
                "faith_failed_turns": 0,
                "tq_total_turns": 0,
                "tq_failed_turns": 0,
                "faithfulness_scores": [],
                "personalization_scores": [],
                "applicability_scores": [],
                "vividness_scores": [],
                "logical_depth_scores": [],
                "pq_total_questions": 0,
                "pq_failed_questions": 0,
                "pq_fitness": [],
                "pq_groundedness": [],
                "pq_diversity": [],
                "pq_answer_quality": [],
                "pq_cross_concept": [],
            },
        )
        g = grouped[backend]
        g["num_eval_files"] += 1
        g["profiles"].add(rec["profile_id"])

        try:
            with open(rec["evaluation_path"], encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            continue

        s = _extract_eval_summary(eval_data)
        g["paired_turns_total"] += s.get("turn_count", {}).get("paired_turns_total", 0) or 0
        g["tutor_turns_total"] += s.get("turn_count", {}).get("tutor_turns_total", 0) or 0

        sf = s.get("source_faithfulness", {})
        tq = s.get("teaching_quality", {})
        g["faith_total_turns"] += sf.get("num_total_turns_total", sf.get("num_total_turns", 0)) or 0
        g["faith_failed_turns"] += sf.get("num_failed_turns_total", sf.get("num_failed_turns", 0)) or 0
        g["tq_total_turns"] += tq.get("num_total_turns_total", tq.get("num_total_turns", 0)) or 0
        g["tq_failed_turns"] += tq.get("num_failed_turns_total", tq.get("num_failed_turns", 0)) or 0

        faith = sf.get("avg_score_overall", sf.get("avg_score"))
        if isinstance(faith, (int, float)):
            g["faithfulness_scores"].append(float(faith))
        personalization = tq.get("avg_personalization_overall", tq.get("personalization", {}).get("avg"))
        if isinstance(personalization, (int, float)):
            g["personalization_scores"].append(float(personalization))
        app = tq.get("avg_applicability_overall", tq.get("applicability", {}).get("avg"))
        if isinstance(app, (int, float)):
            g["applicability_scores"].append(float(app))
        vividness = tq.get("avg_vividness_overall", tq.get("vividness", {}).get("avg"))
        if isinstance(vividness, (int, float)):
            g["vividness_scores"].append(float(vividness))
        logical_depth = tq.get("avg_logical_depth_overall", tq.get("logical_depth", {}).get("avg"))
        if isinstance(logical_depth, (int, float)):
            g["logical_depth_scores"].append(float(logical_depth))

        pq = s.get("practice_questions", {}) or {}
        if pq:
            g["pq_total_questions"] += pq.get("total_questions_across_sessions", pq.get("num_questions", 0)) or 0
            g["pq_failed_questions"] += pq.get("num_eval_failed_questions_total", pq.get("num_eval_failed_questions", 0)) or 0
            for key, lst_key in [
                ("avg_fitness", "pq_fitness"),
                ("avg_groundedness", "pq_groundedness"),
                ("avg_diversity", "pq_diversity"),
                ("avg_answer_quality", "pq_answer_quality"),
                ("avg_cross_concept", "pq_cross_concept"),
            ]:
                v = pq.get(key)
                if isinstance(v, (int, float)):
                    g[lst_key].append(float(v))

    out: dict[str, dict[str, Any]] = {}
    for backend, s in grouped.items():
        s["num_profiles"] = len(s["profiles"])
        backend_summary: dict[str, Any] = {
            "num_eval_files": s["num_eval_files"],
            "num_profiles": s["num_profiles"],
            "paired_turns_total": s["paired_turns_total"],
            "tutor_turns_total": s["tutor_turns_total"],
            "avg_faithfulness": _safe_avg(s["faithfulness_scores"]),
            "faithfulness_eval_failed_turns": s["faith_failed_turns"],
            "faithfulness_eval_total_turns": s["faith_total_turns"],
            "faithfulness_eval_failed_ratio": round(s["faith_failed_turns"] / s["faith_total_turns"], 4) if s["faith_total_turns"] else 0.0,
            "avg_personalization": _safe_avg(s["personalization_scores"]),
            "avg_applicability": _safe_avg(s["applicability_scores"]),
            "avg_vividness": _safe_avg(s["vividness_scores"]),
            "avg_logical_depth": _safe_avg(s["logical_depth_scores"]),
            "teaching_quality_eval_failed_turns": s["tq_failed_turns"],
            "teaching_quality_eval_total_turns": s["tq_total_turns"],
            "teaching_quality_eval_failed_ratio": round(s["tq_failed_turns"] / s["tq_total_turns"], 4) if s["tq_total_turns"] else 0.0,
        }
        if s["pq_total_questions"] > 0:
            backend_summary["practice_questions"] = {
                "total_questions": s["pq_total_questions"],
                "eval_failed_questions": s["pq_failed_questions"],
                "eval_failed_ratio": round(s["pq_failed_questions"] / s["pq_total_questions"], 4) if s["pq_total_questions"] else 0.0,
                "avg_fitness": _safe_avg(s["pq_fitness"]),
                "avg_groundedness": _safe_avg(s["pq_groundedness"]),
                "avg_diversity": _safe_avg(s["pq_diversity"]),
                "avg_answer_quality": _safe_avg(s["pq_answer_quality"]),
                "avg_cross_concept": _safe_avg(s["pq_cross_concept"]),
            }
        out[backend] = backend_summary
    return out


def _collect_eval_file_records(evaluations_root: Path, kb_names: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    records: list[dict[str, str]] = []

    if kb_names:
        kbs = kb_names
    else:
        kbs = sorted(p.name for p in evaluations_root.iterdir() if p.is_dir()) if evaluations_root.exists() else []

    for kb in kbs:
        kb_dir = evaluations_root / kb
        if not kb_dir.exists():
            continue
        for backend_dir in sorted(p for p in kb_dir.iterdir() if p.is_dir()):
            backend = backend_dir.name
            for eval_path in sorted(backend_dir.glob("*_eval.json")):
                profile_id = eval_path.name[:-10] if eval_path.name.endswith("_eval.json") else eval_path.stem
                records.append(
                    {
                        "kb_name": kb,
                        "backend": backend,
                        "profile_id": profile_id,
                        "evaluation_path": str(eval_path),
                    }
                )

    return kbs, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-KB and global evaluation summaries")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--kb-names",
        default="",
        help="Comma-separated KB names. Empty means auto-discover under evaluations/",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    evaluations_root = output_root / "evaluations"
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    kb_filter = _parse_names(args.kb_names) if args.kb_names.strip() else []
    kb_names, all_records = _collect_eval_file_records(evaluations_root, kb_filter)

    by_kb_manifest: dict[str, Any] = {}
    global_records: list[dict[str, str]] = []

    for kb_name in kb_names:
        kb_records = [r for r in all_records if r["kb_name"] == kb_name]
        by_backend = _build_backend_aggregate(kb_records)
        kb_summary = {
            "timestamp": datetime.now().isoformat(),
            "output_root": str(output_root),
            "kb_name": kb_name,
            "num_eval_files": len(kb_records),
            "num_backends": len(by_backend),
            "by_backend": by_backend,
        }
        kb_summary_path = evaluations_root / kb_name / "summary.json"
        kb_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(kb_summary_path, "w", encoding="utf-8") as f:
            json.dump(kb_summary, f, ensure_ascii=False, indent=2)
        kb_summary_md_path = evaluations_root / kb_name / "summary.md"
        kb_md = _build_markdown_report(
            title=f"Evaluation Summary - {kb_name}",
            output_root=output_root,
            num_eval_files=len(kb_records),
            by_backend=by_backend,
        )
        kb_summary_md_path.write_text(kb_md, encoding="utf-8")

        by_kb_manifest[kb_name] = {
            "summary_path": str(kb_summary_path),
            "summary_markdown_path": str(kb_summary_md_path),
            "num_eval_files": len(kb_records),
            "num_backends": len(by_backend),
        }
        global_records.extend(kb_records)
        print(f"[KB] {kb_name}: {kb_summary_path}")
        print(f"[KB Table] {kb_name}: {kb_summary_md_path}")

    overall_by_backend = _build_backend_aggregate(global_records)
    manifest = {
        "step": "eval_summary",
        "timestamp": datetime.now().isoformat(),
        "output_root": str(output_root),
        "kb_names": kb_names,
        "num_kbs": len(kb_names),
        "num_eval_files_total": len(global_records),
        "by_kb": by_kb_manifest,
        "overall": {
            "num_backends": len(overall_by_backend),
            "by_backend": overall_by_backend,
        },
    }
    manifest_path = manifests_root / "eval_summary_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest_md_path = manifests_root / "eval_summary_manifest.md"
    manifest_md = _build_markdown_report(
        title="Evaluation Summary - Overall",
        output_root=output_root,
        num_eval_files=len(global_records),
        by_backend=overall_by_backend,
    )
    manifest_md_path.write_text(manifest_md, encoding="utf-8")

    print(f"[Manifest] {manifest_path}")
    print(f"[Manifest Table] {manifest_md_path}")
    print(
        f"Done. KBs: {manifest['num_kbs']} | "
        f"Eval files: {manifest['num_eval_files_total']} | "
        f"Backends(overall): {manifest['overall']['num_backends']}"
    )


if __name__ == "__main__":
    main()

