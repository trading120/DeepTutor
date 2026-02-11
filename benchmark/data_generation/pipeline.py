#!/usr/bin/env python
"""
Data Generation Pipeline

Orchestrates the full benchmark data generation from existing knowledge bases:

  Existing KBs (per-subtopic)
       │
       ▼  Stage 1: Discover KBs
       │
       ▼  Stage 2: Query each KB → generate knowledge scope
       │
       ▼  Stage 3: Generate student profiles (beginner/intermediate/advanced)
       │
       ▼  Stage 4: Generate knowledge gaps per profile
       │
       ▼  Stage 5: Generate tasks per (profile, gaps) combo
       │
       ▼  Stage 6: Output benchmark dataset (JSONL)
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from benchmark.data_generation.gap_generator import generate_gaps_for_profiles
from benchmark.data_generation.profile_generator import generate_profiles_for_kb
from benchmark.data_generation.scope_generator import generate_knowledge_scope
from benchmark.data_generation.task_generator import generate_tasks

logger = logging.getLogger("benchmark.pipeline")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DataGenerationPipeline:
    """
    Full data generation pipeline: KB → evaluation data.

    Assumes subtopics have already been split and each topic has its own
    RAG knowledge base ready to query.

    Usage:
        pipeline = DataGenerationPipeline()
        entries = await pipeline.run()
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to benchmark_config.yaml.
                         If None, uses benchmark/config/benchmark_config.yaml.
        """
        if config_path is None:
            config_path = PROJECT_ROOT / "benchmark" / "config" / "benchmark_config.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = PROJECT_ROOT / config_path

        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Output directory
        self.output_dir = Path(self.config["output"]["output_dir"])
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # KB base directory
        kb_cfg = self.config.get("knowledge_bases", {})
        self.kb_base_dir = kb_cfg.get("base_dir", "./data/knowledge_bases")
        if not Path(self.kb_base_dir).is_absolute():
            self.kb_base_dir = str(PROJECT_ROOT / self.kb_base_dir)

        # Pipeline state
        self.kb_names: list[str] = []
        self.knowledge_scopes: dict[str, dict] = {}   # kb_name → scope
        self.profiles: dict[str, list[dict]] = {}      # kb_name → [profile]
        self.gaps: dict[str, dict[str, list[dict]]] = {}  # kb_name → {pid → [gap]}
        self.benchmark_entries: list[dict] = []

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    async def run(self, kb_names: list[str] | None = None) -> list[dict]:
        """
        Run the complete data generation pipeline.

        Args:
            kb_names: Explicit list of KB names to process.
                      If None, uses config or auto-discovers.

        Returns:
            List of benchmark entries
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 60)
        logger.info("Starting Benchmark Data Generation Pipeline")
        logger.info("=" * 60)

        # Stage 1: Discover KBs
        await self._stage_discover_kbs(kb_names)

        # Stages 2-5: Process each KB
        for kb_name in self.kb_names:
            await self._process_single_kb(kb_name)

        # Stage 6: Output
        await self._stage_output(timestamp)

        logger.info("=" * 60)
        logger.info(f"Pipeline complete! Generated {len(self.benchmark_entries)} benchmark entries")
        logger.info("=" * 60)

        return self.benchmark_entries

    # =========================================================================
    # Pipeline Stages
    # =========================================================================

    async def _stage_discover_kbs(self, kb_names: list[str] | None = None):
        """Stage 1: Discover which knowledge bases to process."""
        logger.info("[Stage 1] Discovering knowledge bases...")

        if kb_names:
            self.kb_names = kb_names
        else:
            configured = self.config.get("knowledge_bases", {}).get("kb_names", [])
            if configured:
                self.kb_names = configured
            else:
                # Auto-discover from KB manager
                from src.knowledge.manager import KnowledgeBaseManager

                manager = KnowledgeBaseManager(self.kb_base_dir)
                self.kb_names = manager.list_knowledge_bases()

        logger.info(f"Will process {len(self.kb_names)} knowledge bases: {self.kb_names}")

    async def _process_single_kb(self, kb_name: str):
        """Stages 2-5 for a single knowledge base."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing KB: {kb_name}")
        logger.info(f"{'='*50}")

        # Stage 2: Generate knowledge scope
        logger.info(f"[Stage 2] Generating knowledge scope for '{kb_name}'...")
        rag_cfg = self.config.get("rag_query", {})
        try:
            scope = await generate_knowledge_scope(
                kb_name=kb_name,
                seed_queries=rag_cfg.get("seed_queries"),
                mode=rag_cfg.get("mode", "naive"),
                kb_base_dir=self.kb_base_dir,
            )
            self.knowledge_scopes[kb_name] = scope
        except Exception as e:
            logger.error(f"Failed to generate scope for '{kb_name}': {e}")
            return

        # Stage 3: Generate profiles
        logger.info(f"[Stage 3] Generating student profiles for '{kb_name}'...")
        profile_cfg = self.config.get("profile_generation", {})
        profiles = await generate_profiles_for_kb(
            knowledge_scope=scope,
            background_types=profile_cfg.get(
                "background_types", ["beginner", "intermediate", "advanced"]
            ),
            profiles_per_kb=profile_cfg.get("profiles_per_subtopic", 3),
        )
        self.profiles[kb_name] = profiles

        # Stage 4: Generate gaps
        logger.info(f"[Stage 4] Generating knowledge gaps for '{kb_name}'...")
        gap_cfg = self.config.get("gap_generation", {})
        severity_weights = {}
        for level in gap_cfg.get("severity_levels", []):
            severity_weights[level["name"]] = level["weight"]

        profile_gaps = await generate_gaps_for_profiles(
            knowledge_scope=scope,
            profiles=profiles,
            gaps_per_profile=gap_cfg.get("gaps_per_profile", 3),
            severity_weights=severity_weights or None,
        )
        self.gaps[kb_name] = profile_gaps

        # Stage 5: Generate tasks
        logger.info(f"[Stage 5] Generating tasks for '{kb_name}'...")
        task_cfg = self.config.get("task_generation", {})
        task_type_config = [
            {"name": t["name"], "weight": t["weight"]}
            for t in task_cfg.get("task_types", [])
        ] or None
        tasks_per_combo = task_cfg.get("tasks_per_combo", 2)

        for profile in profiles:
            profile_id = profile.get("profile_id", "unknown")
            gaps = profile_gaps.get(profile_id, [])

            if not gaps:
                logger.warning(f"No gaps for profile {profile_id}, skipping tasks")
                continue

            try:
                tasks = await generate_tasks(
                    knowledge_scope=scope,
                    student_profile=profile,
                    knowledge_gaps=gaps,
                    num_tasks=tasks_per_combo,
                    task_type_config=task_type_config,
                )

                # Create benchmark entries: one per task
                for task in tasks:
                    entry = {
                        "kb_name": kb_name,
                        "knowledge_scope": scope,
                        "profile": profile,
                        "gaps": gaps,
                        "task": task,
                    }
                    self.benchmark_entries.append(entry)

            except Exception as e:
                logger.error(f"Failed to generate tasks for {profile_id}: {e}")

        # Save intermediate per-KB result
        if self.config["output"].get("save_intermediate", True):
            self._save_intermediate(
                f"kb_{kb_name}.json",
                {
                    "kb_name": kb_name,
                    "knowledge_scope": scope,
                    "profiles": profiles,
                    "gaps": profile_gaps,
                    "num_entries": sum(
                        1
                        for e in self.benchmark_entries
                        if e.get("kb_name") == kb_name
                    ),
                },
            )

        kb_entries = sum(1 for e in self.benchmark_entries if e["kb_name"] == kb_name)
        logger.info(f"Completed KB '{kb_name}': {len(profiles)} profiles, {kb_entries} entries")

    async def _stage_output(self, timestamp: str):
        """Stage 6: Save final output."""
        logger.info("[Stage 6] Saving benchmark dataset...")

        output_format = self.config["output"].get("format", "jsonl")
        output_file = self.output_dir / f"benchmark_{timestamp}.{output_format}"

        if output_format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for entry in self.benchmark_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        elif output_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.benchmark_entries, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

        logger.info(f"Saved {len(self.benchmark_entries)} entries → {output_file}")

        # Summary
        summary = {
            "timestamp": timestamp,
            "statistics": {
                "total_entries": len(self.benchmark_entries),
                "num_kbs": len(self.kb_names),
                "num_profiles": sum(len(p) for p in self.profiles.values()),
                "per_kb": [
                    {
                        "kb_name": kb,
                        "num_profiles": len(self.profiles.get(kb, [])),
                        "num_entries": sum(
                            1 for e in self.benchmark_entries if e["kb_name"] == kb
                        ),
                    }
                    for kb in self.kb_names
                ],
            },
            "output_file": str(output_file),
        }

        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary → {summary_file}")

    # =========================================================================
    # Utilities
    # =========================================================================

    def _save_intermediate(self, filename: str, data):
        """Save intermediate results to output directory."""
        intermediate_dir = self.output_dir / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        filepath = intermediate_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved intermediate: {filepath}")


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main():
    """CLI entry point for running the data generation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Data Generation: KB → Evaluation Data"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to benchmark config file (default: benchmark/config/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--kb-names",
        nargs="+",
        default=None,
        help="Explicit list of KB names to process (overrides config)",
    )

    args = parser.parse_args()

    pipeline = DataGenerationPipeline(args.config)
    await pipeline.run(kb_names=args.kb_names)


if __name__ == "__main__":
    asyncio.run(main())
