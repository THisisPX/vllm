# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

See [AGENTS.md](AGENTS.md) for contribution policies, duplicate-work checks, and accountability rules. The build/test/lint commands below are summarized from there; the AGENTS.md file is authoritative.

### Installation & Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

### Testing

```bash
uv pip install -r requirements/test/cuda.in          # or cuda.txt on x86_64
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Linting

```bash
pre-commit run --all-files                          # all hooks
pre-commit run ruff-check --all-files               # just ruff
pre-commit run mypy-3.12 --all-files --hook-stage manual  # mypy as in CI
```

Line length: 88 chars. Docstrings: Google-style (`Args:`/`Returns:`/`Raises:`).

### CI Failure Logs

```bash
.buildkite/scripts/ci-fetch-log.sh --pr <PR>
.buildkite/scripts/ci-fetch-log.sh "<buildkite_url>"
```

---

## High-Level Architecture

vLLM is an LLM inference and serving engine built around **PagedAttention** (efficient KV cache memory management). It has ~200+ supported model architectures and runs on NVIDIA, AMD, Intel GPUs, CPUs, and TPUs.

### Dual Engine Architecture

There are **two engine implementations**:

| | V0 (Legacy) | V1 (Current) |
|---|---|---|
| **Engine** | `vllm/engine/llm_engine.py` | `vllm/v1/engine/llm_engine.py` |
| **Async** | `vllm/engine/async_llm_engine.py` | `vllm/v1/engine/async_llm.py` |
| **Core loop** | built into LLMEngine | `vllm/v1/engine/core.py` (`EngineCore`) |
| **Scheduler** | integrated | `vllm/v1/core/sched/scheduler.py` |
| **Model runner** | `vllm/worker/` | `vllm/v1/worker/gpu_model_runner.py` |

V1 is the primary implementation. V0 is kept for backward compatibility. The two don't share much code — V1 is a ground-up rewrite with better separation of concerns.

### V1 Pipeline (Request Lifecycle)

```
User Request → LLMEngine → InputProcessor → EngineCore (via ZMQ/IPC)
                                           ↓
                              Scheduler → KV Cache Manager
                                           ↓
                              Executor → Worker → ModelRunner
                                           ↓
                              Outputs back → OutputProcessor → User
```

1. **`LLMEngine`** (`vllm/v1/engine/llm_engine.py`): Public API. Converts `PromptType` inputs into `EngineCoreRequest` objects. Manages tokenization, output processing, and parallel sampling. Thin wrapper around `EngineCore`.

2. **`EngineCore`** (`vllm/v1/engine/core.py`): The inner loop. Initializes the model executor, KV caches, scheduler, and structured output manager. Runs the step loop: schedule → execute model → process outputs. Runs in its own process or thread (separated via `EngineCoreClient`).

3. **`Scheduler`** (`vllm/v1/core/sched/scheduler.py`): Decides which requests get compute, manages KV cache block allocation, handles preemption. Produces `SchedulerOutput` containing batch composition, block tables, and token budgets.

4. **`Executor`** (`vllm/v1/executor/abstract.py`): Abstraction over single-GPU (`UniProcExecutor`), multi-process (`MultiprocExecutor`), or Ray-based (`RayExecutor`) execution. Owns the worker(s).

5. **`GPUModelRunner`** (`vllm/v1/worker/gpu_model_runner.py`): ~4000-line workhorse. Prepares input tensors, manages KV cache slot mapping, runs the forward pass, handles CUDA graphs, and produces `ModelRunnerOutput`.

### Configuration System

`VllmConfig` (`vllm/config/vllm.py`) is the master configuration dataclass that composes ~25 sub-configs:

- `ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig`
- `DeviceConfig`, `LoadConfig`, `LoRAConfig`, `SpeculativeConfig`
- `CompilationConfig`, `AttentionConfig`, `KernelConfig`, `QuantizationConfig`
- `MultiModalConfig`, `StructuredOutputsConfig`, `ReasoningConfig`
- And more: `OffloadConfig`, `KVTransferConfig`, `ECTransferConfig`, `KVEventsConfig`, `MambaConfig`, `PoolerConfig`, `SpeechToTextConfig`, `DiffusionConfig`, `ProfilerConfig`, `WeightTransferConfig`

Engine args (`vllm/engine/arg_utils.py`) parse CLI args and construct `VllmConfig`. Use `EngineArgs` (offline/sync) or `AsyncEngineArgs` (server).

### Model Registry

Models are registered in `vllm/model_executor/models/registry.py` via the `_TEXT_GENERATION_MODELS` dict, which maps HuggingFace `config.architectures` class names to `(module_file, class_name)` tuples:

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
"DeepseekV3ForCausalLM": ("deepseek_v3", "DeepseekV3ForCausalLM"),
```

Each model module lives in `vllm/model_executor/models/` and typically implements:
- A `*ForCausalLM` class inheriting from `VllmModelForTextGeneration`
- Model-specific attention, MLP, or MoE layers in `vllm/model_executor/layers/`

### Key Subsystems

**KV Cache** (`vllm/v1/core/kv_cache_*.py`): PagedAttention block management. `KVCacheManager` allocates/frees blocks, `BlockPool` manages the block pool, `KVCacheCoordinator` handles hybrid (prefix + streaming) KV cache. Blocks are sized per `block_size` (default 16 tokens).

**Attention** (`vllm/v1/attention/`): Backends include FlashAttention, FlashInfer, MLA, Triton, and custom CUDA kernels. Configured via `AttentionConfig`.

**Distributed** (`vllm/distributed/`): Parallelism strategies:
- **TP** (tensor parallelism) — splits weight tensors across GPUs
- **PP** (pipeline parallelism) — splits layers across GPUs
- **DP** (data parallelism) — replicates model across GPUs
- **EP** (expert parallelism) — distributes MoE experts
- **DCP** (disaggregated prefill/decode) — separates prefill and decode nodes

Set via `ParallelConfig`. Communication coordinated through `parallel_state.py`.

**Quantization** (`vllm/model_executor/layers/quantization/`): FP8, INT8, INT4, GPTQ, AWQ, GGUF, compressed-tensors, TPU-int8, and more. Configured via `QuantizationConfig`.

**Multi-Modal** (`vllm/multimodal/`): Handles images, video, and audio inputs. `MULTIMODAL_REGISTRY` maps model types to their modality processors.

**Compilation** (`vllm/compilation/`): CUDA Graph capture (full and piecewise), torch.compile integration, breakable CUDA graphs. `CompilationConfig` controls mode (`NONE`, `FULL`, `PIECEWISE`, `VLLM_COMPILE`).

**Structured Output** (`vllm/v1/structured_output/`): Grammar-constrained generation via xgrammar, guidance, or lm-format-enforcer.

**Speculative Decoding** (`vllm/v1/spec_decode/`): Draft-model-based generation (EAGLE, n-gram, suffix). Configured via `SpeculativeConfig`.

**LoRA** (`vllm/lora/`): Low-Rank Adapter support for dense and MoE layers with dynamic loading/unloading.

**KV Transfer / Disaggregated Serving** (`vllm/distributed/kv_transfer/`): Transfers KV cache from prefill servers to decode servers for disaggregated architectures.

**Rust Frontend** (`rust/src/`): A Rust-based HTTP server (`server`), tokenizer (`tokenizer`), chat renderer (`chat`), tool parser (`tool-parser`), reasoning parser (`reasoning-parser`), and engine-core-client bridge. The Rust server can replace the Python FastAPI server. Python ↔ Rust interoperability via PyO3 and ZMQ.

### Entrypoints

- **CLI** (`vllm/entrypoints/cli/`): `vllm serve`, `vllm chat`, `vllm bench`, `vllm run-batch`. Entry point: `pyproject.toml` → `vllm.entrypoints.cli.main:main`.
- **OpenAI API** (`vllm/entrypoints/openai/`): `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, etc.
- **Python API**: `vllm.LLM` (`vllm/entrypoints/llm.py`) for offline inference.
- **Anthropic API** (`vllm/entrypoints/anthropic/`): Anthropic Messages API support.
- **gRPC** (`vllm/entrypoints/serve/`): Optional gRPC server.
- **Pooling** (`vllm/entrypoints/pooling/`): Embedding, classification, scoring.

### Plugin System

Plugins are loaded via `importlib.metadata` entry points under `vllm.general_plugins`. The built-in plugins are LoRA filesystem and HuggingFace Hub resolvers (`vllm/plugins/`).

### Environment Variables

All env vars are declared at the top of `vllm/envs.py` as typed module-level attributes inside a `TYPE_CHECKING` block. The `vllm/env_override.py` module imports first (before any other vllm module in `__init__.py`) to apply overrides. Key env vars: `VLLM_TARGET_DEVICE`, `VLLM_USE_PRECOMPILED`, `VLLM_USE_PRECOMPILED_RUST`, `MAX_JOBS`, `NVCC_THREADS`.

### Build System

`setup.py` uses CMake for C++/CUDA extensions and setuptools-rust for Rust extensions. Key build details:
- `VLLM_TARGET_DEVICE`: `cuda`, `rocm`, `xpu`, `cpu`, `tpu`, or `empty`
- `VLLM_USE_PRECOMPILED=1`: downloads precompiled wheels from `wheels.vllm.ai`
- Extension targets are defined in `setup.py` based on device type (CUDA extensions like `_vllm_fa2_C`, `_vllm_fa3_C`, `_C`, `_moe_C`, etc.)
- Rust frontend binaries ship as precompiled `.so` files alongside the Python package

### Test Organization

```
tests/
├── models/           # Model-specific tests (language, multimodal, pooling)
├── v1/               # V1 engine tests (scheduler, executor, core, etc.)
├── entrypoints/      # API server tests
├── distributed/      # Distributed inference tests
├── lora/             # LoRA tests
├── kernels/          # CUDA kernel tests
├── quantization/     # Quantization tests
├── sampling/         # Sampling parameter tests
├── plugins/          # Plugin system tests
├── tokenizers/       # Tokenizer tests
├── compile/          # Compilation system tests
└── benchmarks/       # Benchmark tests
```

Important pytest markers: `core_model`, `hybrid_model`, `cpu_model`, `cpu_test`, `split`, `distributed`, `optional`, `slow_test`.
