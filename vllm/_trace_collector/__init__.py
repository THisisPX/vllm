# SPDX-License-Identifier: Apache-2.0
"""Expert Trace Collection for MoE PD Disaggregation Research.

This package provides a low-overhead mechanism to record which experts
are selected for each token during MoE inference, enabling offline
analysis of expert locality and cold-start effects.

Public API
----------
- ``ExpertTraceCollector`` — primary class; attach to GPU model runner.
- ``TraceContext`` — per-engine-step metadata passed into the forward context.
- ``TraceWriter`` — thread-safe JSONL writer (used internally).
- ``RequestMetadata`` — per-request metadata for analysis.
"""

from .collector import ExpertTraceCollector
from .context import RequestMetadata, TraceContext
from .writer import TraceWriter

__all__ = [
    "ExpertTraceCollector",
    "TraceContext",
    "TraceWriter",
    "RequestMetadata",
]
