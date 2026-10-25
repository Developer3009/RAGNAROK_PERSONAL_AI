"""
ragnarok/eval/__init__.py
────────────────────────
Phase 8: Evaluation & Benchmark Framework for Ragnarok Local AI Agent.
"""

from ragnarok.eval.datasets import (
    SINGLE_STEP_DATASET,
    MULTI_STEP_DATASET,
    MEMORY_RECALL_DATASET,
    SAFETY_INJECTION_DATASET,
    EvalPrompt,
    MultiStepGoal,
    MemoryRecallCase,
    SafetyThreatCase,
)
from ragnarok.eval.benchmarks import (
    BenchmarkSuite,
    LatencyBenchmark,
    ResourceBenchmark,
    AccuracyBenchmark,
    BenchmarkResult,
)
from ragnarok.eval.runner import EvaluationRunner

__all__ = [
    "SINGLE_STEP_DATASET",
    "MULTI_STEP_DATASET",
    "MEMORY_RECALL_DATASET",
    "SAFETY_INJECTION_DATASET",
    "EvalPrompt",
    "MultiStepGoal",
    "MemoryRecallCase",
    "SafetyThreatCase",
    "BenchmarkSuite",
    "LatencyBenchmark",
    "ResourceBenchmark",
    "AccuracyBenchmark",
    "BenchmarkResult",
    "EvaluationRunner",
]
