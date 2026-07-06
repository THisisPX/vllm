# SPDX-License-Identifier: Apache-2.0
"""Expert Activation Profile (EAP) — lightweight expert frequency metadata.

EAP captures per-request expert activation statistics during Prefill
and transfers them alongside KV Cache to the Decode pool, where they
guide expert cache initialization and replacement policy.

Components
----------
- ``EapProfile`` — L×E frequency matrix, serializable (~6.5 KB for DSv2-Lite)
- ``EapAccumulator`` — per-request prefill histogram collector
- ``EapCacheManager`` — EAP-weighted LRU cache for decode
"""

from .profile import EapProfile
from .accumulator import EapAccumulator
from .cache_manager import EapCacheManager
from .integration import EapIntegration, create_from_env

__all__ = [
    "EapProfile",
    "EapAccumulator",
    "EapCacheManager",
    "EapIntegration",
    "create_from_env",
]
