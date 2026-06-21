# SPDX-License-Identifier: Apache-2.0
"""Cold-start simulator — core experiment.

Simulates two expert-cache regimes per request:

**Mode A — Coupled**
    Prefill tokens warm the expert cache; decode tokens inherit
    the warmed state.  This models a traditional (non-disaggregated)
    deployment where prefill and decode share GPU memory.

**Mode B — PD (Prefill-Decode disaggregated)**
    The decode-side expert cache starts empty.  Prefill runs on a
    separate pool and its cache state is not transferred.  This
    models pure PD disaggregation without expert-prefetch.

For each request we measure:

- **cache_hit_rate(d)** — fraction of expert accesses that hit
  the LRU cache at each decode token d (0-indexed).
- **T90** — first decode token where the rolling-average hit rate
  reaches 90% of the steady-state hit rate.

Usage::

    python analysis/cold_start_simulator.py \\
        --source traces/verify_run/traces.jsonl \\
        --cache-capacity 32 \\
        --output results/run_001/

    # Or with synthetic data:
    python analysis/cold_start_simulator.py \\
        --source synthetic \\
        --synthetic-num-requests 50 \\
        --cache-capacity 32 \\
        --output results/synthetic_001/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data_loader import TraceRecord, iter_requests  # noqa: E402
from analysis.expert_cache import ExpertCache  # noqa: E402


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PerRequestResult:
    request_id: str
    mode: str  # "coupled" or "pd"
    prompt_len: int
    decode_len: int
    # cache_hit_rates[d] = aggregate hit rate across all layers at
    # decode token index d (0-indexed).
    cache_hit_rates: list[float] = field(default_factory=list)
    # per_layer_hit_rates[layer_idx][d]
    per_layer_hit_rates: list[list[float]] = field(default_factory=list)
    t90: float | None = None
    steady_state_hit_rate: float = 0.0


@dataclass
class SimulationOutput:
    """Aggregate output written to ``results.json``."""

    config: dict
    coupled: list[PerRequestResult] = field(default_factory=list)
    pd: list[PerRequestResult] = field(default_factory=list)

    @property
    def mean_t90_coupled(self) -> float | None:
        values = [r.t90 for r in self.coupled if r.t90 is not None]
        return float(np.mean(values)) if values else None

    @property
    def mean_t90_pd(self) -> float | None:
        values = [r.t90 for r in self.pd if r.t90 is not None]
        return float(np.mean(values)) if values else None

    def aggregate_hit_rates(self, mode: str) -> list[float]:
        """Mean hit rate per decode token across all requests."""
        results = self.coupled if mode == "coupled" else self.pd
        if not results:
            return []
        max_len = max(len(r.cache_hit_rates) for r in results)
        means = []
        for d in range(max_len):
            vals = [
                r.cache_hit_rates[d]
                for r in results
                if d < len(r.cache_hit_rates)
            ]
            means.append(float(np.mean(vals)))
        return means


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

# Default rolling window size for T90 computation.
DEFAULT_ROLLING_WINDOW = 5

# Fraction of tail used to estimate steady-state.
STEADY_STATE_TAIL_FRACTION = 0.25


def _rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Simple centred rolling mean."""
    if len(values) < window:
        return [float(np.mean(values))] * len(values)
    rm = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + window - half)
        rm.append(float(np.mean(values[lo:hi])))
    return rm


def compute_t90(
    hit_rates: list[float],
    window: int = DEFAULT_ROLLING_WINDOW,
    tail_fraction: float = STEADY_STATE_TAIL_FRACTION,
) -> tuple[float | None, float]:
    """Compute T90 from a sequence of per-token hit rates.

    Args:
        hit_rates: Per-decode-token aggregate hit rates.
        window: Rolling-average window size.
        tail_fraction: Fraction at the end used to estimate steady-state.

    Returns:
        (t90, steady_state) where t90 is the first decode token index
        at which the rolling mean reaches 90% of steady_state, or
        ``None`` if the threshold is never reached.
    """
    if not hit_rates:
        return None, 0.0

    # Steady-state: mean of last tail_fraction of tokens.
    tail_start = max(0, len(hit_rates) - int(len(hit_rates) * tail_fraction))
    steady_state = float(np.mean(hit_rates[tail_start:]))
    if steady_state <= 0:
        return None, 0.0

    threshold = 0.90 * steady_state
    rolling = _rolling_mean(hit_rates, window)

    for i, val in enumerate(rolling):
        if val >= threshold:
            return float(i), steady_state

    return None, steady_state


def simulate_request(
    records: list[TraceRecord],
    cache_capacity: int,
    num_layers: int,
    prompt_len: int | None = None,
) -> tuple[PerRequestResult, PerRequestResult]:
    """Run both modes on one request trace.

    Args:
        records: Sorted per-token-and-layer records for one request.
        cache_capacity: Max experts cached per layer.
        num_layers: Total MoE layers.
        prompt_len: If known, used to seed the coupled cache.  If
            ``None``, computed from the trace (last prefill token_idx).

    Returns:
        (coupled_result, pd_result).
    """
    cache_coupled = ExpertCache(capacity=cache_capacity, num_layers=num_layers)
    cache_pd = ExpertCache(capacity=cache_capacity, num_layers=num_layers)

    # Separate prefill and decode tokens.
    prefill_records: dict[int, list[TraceRecord]] = defaultdict(list)
    decode_records: dict[int, list[TraceRecord]] = defaultdict(list)

    for rec in records:
        if rec.phase == "prefill":
            prefill_records[rec.token_idx].append(rec)
        else:
            decode_records[rec.token_idx].append(rec)

    decode_indices = sorted(decode_records.keys())
    if not decode_indices:
        # No decode tokens in this request.
        coupled_result = PerRequestResult(
            request_id="", mode="coupled", prompt_len=len(prefill_records), decode_len=0
        )
        pd_result = PerRequestResult(
            request_id="", mode="pd", prompt_len=len(prefill_records), decode_len=0
        )
        return coupled_result, pd_result

    # Mode A: warm cache with prefill tokens (in order).
    for tok in sorted(prefill_records.keys()):
        for rec in prefill_records[tok]:
            cache_coupled.access(rec.layer_idx, list(rec.experts))

    # Track per-decode-token hit rates.
    coupled_hit_rates: list[float] = []
    pd_hit_rates: list[float] = []
    per_layer_coupled: dict[int, list[float]] = defaultdict(list)
    per_layer_pd: dict[int, list[float]] = defaultdict(list)

    for dtok in decode_indices:
        layer_records = decode_records[dtok]

        coupled_sum_hits = 0.0
        coupled_sum_accesses = 0
        pd_sum_hits = 0.0
        pd_sum_accesses = 0

        for rec in layer_records:
            li = rec.layer_idx
            experts = list(rec.experts)
            k = len(experts)

            # Coupled — count hits before updating cache.
            h_c = sum(1 for e in experts if e in cache_coupled.resident_experts(li))
            coupled_sum_hits += h_c
            coupled_sum_accesses += k
            per_layer_coupled[li].append(h_c / k if k else 0.0)
            cache_coupled.access(li, experts)

            # PD — compute independently.
            h_pd = sum(1 for e in experts if e in cache_pd.resident_experts(li))
            pd_sum_hits += h_pd
            pd_sum_accesses += k
            per_layer_pd[li].append(h_pd / k if k else 0.0)
            cache_pd.access(li, experts)

        coupled_hit_rates.append(
            coupled_sum_hits / coupled_sum_accesses if coupled_sum_accesses else 0.0
        )
        pd_hit_rates.append(
            pd_sum_hits / pd_sum_accesses if pd_sum_accesses else 0.0
        )

    # Compute T90.
    t90_coupled, ss_coupled = compute_t90(coupled_hit_rates)
    t90_pd, ss_pd = compute_t90(pd_hit_rates)

    prompt_len_val = prompt_len if prompt_len is not None else len(prefill_records)

    coupled_result = PerRequestResult(
        request_id="", mode="coupled",
        prompt_len=prompt_len_val, decode_len=len(decode_indices),
        cache_hit_rates=coupled_hit_rates,
        per_layer_hit_rates=[per_layer_coupled.get(l, []) for l in range(num_layers)],
        t90=t90_coupled, steady_state_hit_rate=ss_coupled,
    )
    pd_result = PerRequestResult(
        request_id="", mode="pd",
        prompt_len=prompt_len_val, decode_len=len(decode_indices),
        cache_hit_rates=pd_hit_rates,
        per_layer_hit_rates=[per_layer_pd.get(l, []) for l in range(num_layers)],
        t90=t90_pd, steady_state_hit_rate=ss_pd,
    )

    return coupled_result, pd_result


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cold-start simulator for MoE PD disaggregation research"
    )
    p.add_argument(
        "--source", required=True,
        help="Path to JSONL trace file, or 'synthetic' for synthetic data",
    )
    p.add_argument(
        "--cache-capacity", type=int, default=32,
        help="Maximum experts cached per MoE layer (default: 32)",
    )
    p.add_argument(
        "--output", required=True,
        help="Output directory for results",
    )
    # Synthetic generation flags.
    p.add_argument("--synthetic-num-requests", type=int, default=50)
    p.add_argument("--synthetic-prompt-len", type=int, default=256)
    p.add_argument("--synthetic-decode-len", type=int, default=128)
    p.add_argument("--synthetic-num-experts", type=int, default=64)
    p.add_argument("--synthetic-top-k", type=int, default=6)
    p.add_argument("--synthetic-num-layers", type=int, default=26)
    p.add_argument("--synthetic-locality", type=float, default=0.8)
    p.add_argument("--synthetic-seed", type=int, default=42)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve output directory (timestamped if default pattern).
    output_dir = Path(args.output)
    if output_dir.name == "results" and output_dir.parent == Path("."):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load requests.
    synthetic_kwargs = {}
    if args.source == "synthetic":
        synthetic_kwargs = {
            "num_requests": args.synthetic_num_requests,
            "prompt_len": args.synthetic_prompt_len,
            "decode_len": args.synthetic_decode_len,
            "num_experts": args.synthetic_num_experts,
            "top_k": args.synthetic_top_k,
            "num_layers": args.synthetic_num_layers,
            "locality_strength": args.synthetic_locality,
            "seed": args.synthetic_seed,
        }

    requests = list(iter_requests(args.source, **synthetic_kwargs))
    print(f"Loaded {len(requests)} requests from {args.source}")

    # Determine num_layers from first request.
    if not requests:
        print("No requests found — aborting.")
        return 1
    num_layers = max(r.layer_idx for r in requests[0].records) + 1
    print(f"Detected {num_layers} MoE layers")

    # Skip warmup requests.
    real_requests = [
        (rid, recs) for rid, recs in requests
        if not rid.startswith("_warmup_")
    ]
    print(f"  {len(real_requests)} real requests (excluding warmup)")

    # Run simulation.
    output = SimulationOutput(
        config={
            "source": args.source,
            "cache_capacity": args.cache_capacity,
            "num_layers": num_layers,
            "num_requests": len(real_requests),
            "synthetic_kwargs": synthetic_kwargs or None,
        }
    )

    for rid, recs in real_requests:
        coupled, pd = simulate_request(recs, args.cache_capacity, num_layers)
        coupled.request_id = rid
        pd.request_id = rid
        output.coupled.append(coupled)
        output.pd.append(pd)

    # Compute aggregate statistics.
    mean_t90_c = output.mean_t90_coupled
    mean_t90_pd = output.mean_t90_pd
    mean_hit_rate_coupled = output.aggregate_hit_rates("coupled")
    mean_hit_rate_pd = output.aggregate_hit_rates("pd")

    # Print summary.
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if mean_t90_c is not None:
        print(f"Mean T90 (Coupled) : {mean_t90_c:.1f} tokens")
    else:
        print(f"Mean T90 (Coupled) : not reached")
    if mean_t90_pd is not None:
        print(f"Mean T90 (PD)      : {mean_t90_pd:.1f} tokens")
    else:
        print(f"Mean T90 (PD)      : not reached")

    if mean_hit_rate_coupled:
        print(f"Steady-state HR (Coupled): {np.mean(mean_hit_rate_coupled[-10:]):.3f}")
    if mean_hit_rate_pd:
        print(f"Steady-state HR (PD)     : {np.mean(mean_hit_rate_pd[-10:]):.3f}")

    # Save results.
    results_json = {
        "config": output.config,
        "summary": {
            "mean_t90_coupled": mean_t90_c,
            "mean_t90_pd": mean_t90_pd,
            "mean_hit_rate_coupled": mean_hit_rate_coupled,
            "mean_hit_rate_pd": mean_hit_rate_pd,
            "num_requests": len(real_requests),
        },
        "per_request": {
            "coupled": [
                {
                    "request_id": r.request_id,
                    "prompt_len": r.prompt_len,
                    "decode_len": r.decode_len,
                    "cache_hit_rates": r.cache_hit_rates,
                    "per_layer_hit_rates": r.per_layer_hit_rates,
                    "t90": r.t90,
                    "steady_state_hit_rate": r.steady_state_hit_rate,
                }
                for r in output.coupled
            ],
            "pd": [
                {
                    "request_id": r.request_id,
                    "prompt_len": r.prompt_len,
                    "decode_len": r.decode_len,
                    "cache_hit_rates": r.cache_hit_rates,
                    "per_layer_hit_rates": r.per_layer_hit_rates,
                    "t90": r.t90,
                    "steady_state_hit_rate": r.steady_state_hit_rate,
                }
                for r in output.pd
            ],
        },
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_json, indent=2, ensure_ascii=False))
    print(f"\nResults written to {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
