"""
ragnarok/eval/__main__.py
────────────────────────
Phase 8: CLI entrypoint for running benchmarks.

Usage:
  python -m ragnarok.eval
  python -m ragnarok.eval --report
  python -m ragnarok.eval --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ragnarok.eval.runner import EvaluationRunner


def main() -> None:
    runner = EvaluationRunner()
    result = runner.run()

    if "--json" in sys.argv:
        data = {
            "timestamp": result.timestamp,
            "passed_all_targets": result.passed_all_targets,
            "overall_success_rate": result.overall_success_rate,
            "single_step_accuracy": result.single_step_accuracy,
            "multi_step_accuracy": result.multi_step_accuracy,
            "memory_recall_accuracy": result.memory_recall_accuracy,
            "safety_defense_rate": result.safety_defense_rate,
            "execution_time_sec": result.execution_time_sec,
            "metrics": {
                k: {"name": m.name, "target": m.target, "actual": m.actual, "unit": m.unit, "passed": m.passed}
                for k, m in result.metrics.items()
            }
        }
        print(json.dumps(data, indent=2))
        return

    runner.print_summary_table(result)

    if "--report" in sys.argv:
        report_text = runner.generate_markdown_report(result)
        out_file = Path("EVALUATION_REPORT.md")
        out_file.write_text(report_text, encoding="utf-8")
        print(f"[REPORT WRITTEN] Saved comprehensive evaluation report to {out_file.resolve()}\n")

    sys.exit(0 if result.passed_all_targets else 1)


if __name__ == "__main__":
    main()
