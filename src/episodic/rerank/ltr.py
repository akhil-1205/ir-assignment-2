"""Stub for an optional learning-to-rank reranker.

Out of scope for v1. The feature shape below documents what a future
trainer would consume so the rest of the pipeline can plug it in
without restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LTRFeatures:
    bm25_score: float
    dense_state_score: float
    dense_plan_score: float
    tool_overlap: float
    outcome_match: float
    age_seconds: float
