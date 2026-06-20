# Expert Trace Collector (Phase 0)

Collects MoE expert activation traces from vLLM inference for PD disaggregation research.

## Architecture

### Hooks Added

Three minimal changes to vLLM core (approximately 40 lines total):

| File | Change | Lines |
|------|--------|-------|
| `vllm/forward_context.py` | Added `expert_trace_callback` field to `ForwardContext`; accept & pass through in `set_forward_context()` and `create_forward_context()`; conditionally populate `all_moe_layers` when tracing is enabled. | ~15 |
| `vllm/model_executor/layers/fused_moe/router/fused_moe_router.py` | Invoke `ctx.expert_trace_callback(layer_name, topk_ids)` at the end of `FusedMoERouter.select_experts()`, after `_select_experts()` returns final expert assignments. | ~8 |
| `vllm/v1/worker/gpu_model_runner.py` | Check `VLLM_EXPERT_TRACE_DIR` env var; initialize `ExpertTraceCollector`; build `TraceContext` with per-token metadata before each forward pass; pass callback into `set_forward_context()`. | ~35 |

All trace logic lives in `trace_collector/` — vLLM core only provides the interception point.

### Where Expert Selection Is Intercepted

```
GPUModelRunner.execute_model()
  → build TraceContext (per-step per-token metadata)
  → set_forward_context(..., expert_trace_callback)
    → model.forward()
      → MoELayer.forward()
        → router.select_experts(hidden_states, router_logits)
          → _select_experts()  # normal routing, returns topk_ids
          → ctx.expert_trace_callback(layer_name, topk_ids)  # ← HOOK
        → expert computation (unchanged)
```

The hook fires **after** expert selection but before the actual expert computation. This ensures we record the true expert assignment (including any EPLB remapping).

### TraceContext Semantics

`TraceContext` is constructed per engine step with these per-token arrays:

| Field | Length | Description |
|-------|--------|-------------|
| `request_ids` | `num_reqs` | Request IDs in batch order |
| `req_indices` | `num_tokens` | Which request owns each token |
| `is_prefill` | `num_tokens` | `True` if token's request is still in prefill |
| `token_indices` | `num_tokens` | Absolute position in sequence (0-based) |
| `token_ids` | `num_tokens` | Vocabulary token ID (-1 if not available) |
| `decode_token_indices` | `num_tokens` | Position relative to decode start; -1 for prefill tokens |

### Correctness Under Chunked Prefill

Chunked prefill splits a long prompt into multiple forward passes. Our `token_indices` use the formula:

```
positions_np = num_computed_tokens_cpu[req_indices] + query_pos.np
```

This gives correct **absolute** sequence positions even when a request's prefill is split across multiple steps. The `is_prefill` flag is `True` for all chunks until `num_computed_tokens >= num_prompt_tokens`.

### Correctness Under Continuous Batching

Each engine step may include tokens from multiple requests in different phases (some prefill, some decode). The `req_indices` array correctly maps each token to its owning request, and `is_prefill` is computed per-token based on that request's state.

### Correctness Under Mixed Prefill/Decode

Some requests may be in prefill phase while others are in decode phase within the same batch. `is_prefill` is stored per-token, so no ambiguity exists.

## Usage

### Enable Trace Collection

Set the environment variable before starting vLLM:

```bash
export VLLM_EXPERT_TRACE_DIR="./traces/experiment_001"
vllm serve deepseek-ai/DeepSeek-V2-Lite
```

Or in Python:

```python
import os
os.environ["VLLM_EXPERT_TRACE_DIR"] = "./traces/experiment_001"

from vllm import LLM
llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite")
# ... run inference ...
```

### Output Files

```
traces/experiment_001/
├── traces.jsonl      # One JSON record per (token, layer) pair
└── metadata.json     # Layer mapping + request-level metadata
```

### Trace Record Format (`traces.jsonl`)

```json
{
  "request_id": "17",
  "phase": "decode",
  "token_idx": 2048,
  "token_id": 1234,
  "decode_token_idx": 0,
  "layer_idx": 27,
  "experts": [4, 29],
  "step": 42,
  "timestamp": 1712345678.234
}
```

### Metadata Format (`metadata.json`)

```json
{
  "layer_mapping": {
    "0": "model.layers.0.mlp",
    "1": "model.layers.1.mlp"
  },
  "num_layers": 28,
  "num_steps": 1500,
  "total_trace_records": 250000,
  "requests": {
    "17": {
      "prompt_length": 2048,
      "total_decode_tokens": 512
    }
  }
}
```

## Reproducing

1. Install vLLM from source with the trace collector modifications.
2. Choose a MoE model (e.g. `deepseek-ai/DeepSeek-V2-Lite`).
3. Set `VLLM_EXPERT_TRACE_DIR` to the desired output directory.
4. Run inference with your workload.
5. Traces and metadata are written automatically; final flush occurs at engine shutdown.
