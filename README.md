# Qwen Ultrafast

A Rust-hosted SM120 inference project for **Qwen3.8-Flash-Next** on RTX 5090 GPUs. It follows the same broad execution model as DeepSeek Ultrafast: Rust owns CUDA resources and execution, while compiled CUDA C++ kernels implement model computation. B12X is not used by the compute path.

The target checkpoint is [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), with a text-only decoder and routed experts distributed across eight GPUs. This is a focused inference engine, not a general Qwen runtime.

## Current status

The following component tests have run on a real RTX 5090 with CUDA 13.3:

- BF16 and NVFP4 W4A16/W4A4 GEMV.
- NVFP4 activation quantization, byte-exact against FlashInfer on the test fixture.
- Fused expert gate/up/SwiGLU and down/accumulation, bit-exact against split kernels.
- Fused four-stream RMSNorm, bit-exact against separate launches.
- Gated DeltaNet causal convolution and FP32 recurrent state update.
- QSA single-token attention with output gating.
- PLE token hashing and dilated convolution.
- Router softmax and top-10 selection.
- Direct and captured-graph GDN recurrence.

The component suite passes `cargo clippy --workspace --all-targets -- -D warnings`, `compute-sanitizer --tool memcheck`, and `compute-sanitizer --tool racecheck`, with zero sanitizer errors/hazards after fixes.

A small single-layer safetensors fixture has also run through the complete Runtime on one, two, and eight GPUs, including tokenizer input, GDN, hyper-connections, MoE, LM head, and greedy token output. All ranks produced token `24251` for the fixture's first decode step. NCCL's fixed-input allreduce returned the expected values on both GPUs. A two-rank, three-token run also completed with all model computation and NCCL collectives captured in one reusable CUDA Graph. The fixture is an execution test, not a substitute for official-model numerical validation.

The complete 48-layer forward and official checkpoint mapping load successfully on all eight GPUs, with 17.13 GiB of resident weights reported on rank zero. Direct execution and a full NCCL + CUDA Graph produce the same 16 official-model tokens, `[11, 353, 2688, 264, 1496, 2812, 3712, 10756, 852, 579, 7157, 279, 1483, 35442, 1578, 4313]`, decoded as `, I'm a high school history teacher who's spent the last fifteen years trying`. A first-layer, real-weight Transformers comparison localizes and verifies the GDN `sigmoid` output gate: the gated recurrence square sums are 32.1577 (Rust) and 32.1585 (reference). SGLang 0.5.21 with deterministic inference produced exactly the same 16 tokens on four consecutive runs. At 64 tokens, the first 25 tokens match before the implementations diverge, so long-sequence numerical parity is not yet established.

The current fused W4A16 + FP32 NCCL implementation measures **72.56–75.64 tokens/s** in two independent 64-token graph runs with `NCCL_ALGO=TREE`, exceeding deterministic SGLang's **53–54 tokens/s** with decode graph and RING for the same checkpoint and prompt. Replacing dynamic FP4 lookup tables with exact integer-bit decoding removed the main expert-kernel bottleneck. Parallel router selection, fused four-stream RMSNorm, and parallel GDN state updates provide additional gains while preserving the token sequence. The experimental `--w4a4` path does not eliminate the token divergence, so W4A16 remains the default. Layer-by-layer traces through the first 25 steps show smooth numerical accumulation rather than an abrupt cache/state failure; dense-kernel reduction order and quantization differences remain. Kernel timelines show substantial FP32 AllReduce waiting, but their durations also include rank imbalance and should not be interpreted as pure network transfer time. BF16 reduction and a four-rank configuration were tested and rejected because they did not provide valid, faster execution. PLE row lookup remains outside the graph; the rest of the model and NCCL reduction can be captured.

## Target architecture

- 48 decoder layers: 36 Gated DeltaNet and 12 sparse-attention layers.
- Four hyper-connection streams, hidden size 2560.
- 512 routed experts per layer, top 10 per token, plus a shared expert.
- Official NVFP4 routed experts, executed as fused W4A16 by default with an optional W4A4 activation path.
- Official FP8 PLE table with a scalar BF16 scale; rows are fetched on demand.
- Dense weights replicated per rank and routed experts sharded by `expert_id % world_size`.
- Greedy decoding on GPU with stable device buffers.

The initial execution limit is 2048 tokens. At this length QSA selects every complete block and the tail, so dense attention is equivalent to the model's sparse selection. Longer contexts require the QSA indexer and are rejected rather than silently run through dense attention.

Vision, learned MTP, long-context QSA, continuous batching, and the OpenAI-compatible server are not implemented yet.

## Requirements

- RTX 5090, SM120. The CUDA wrapper rejects other compute capabilities.
- CUDA 13 toolkit with `nvcc`.
- Rust 1.85 or newer.
- The official checkpoint in `models/Qwen3.8-Flash-Next-NVFP4`, or pass `--model` with its location.

## Build and component validation

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cargo build --release --workspace
cargo run --release -p qwen-ultrafast-engine --bin test -- --device=0
cargo clippy --workspace --all-targets -- -D warnings
compute-sanitizer --tool memcheck ./target/release/test --device=0
compute-sanitizer --tool racecheck ./target/release/test --device=0
```

The latest component report contains zero error for NVFP4 W4A16/W4A4 GEMV, activation quantization, fused experts, convolution, QSA single-token attention, PLE, and router selection. Fused and split experts are also checked for bit-exact outputs over multiple accumulations. GDN recurrence differs from the CPU reference by approximately `5.96e-8` maximum absolute error after matching canonical BF16 rounding. The PLE hash test also covers signed-int64 remainder semantics after wrapping multiplication.

For direct-mode layer diagnostics, `QWEN_TRACE_VALUES=1` prints stage square sums, the first eight values, and top-10 logits on rank zero. `QWEN_UNFUSED_MOE=1` selects the split-expert A/B path. `QWEN_NVFP4_DUMP=/path/to/fixture.json` exports the quantization fixture for comparison with FlashInfer. `python/hf_layer0_reference.py` independently loads the real first layer with Transformers and dequantizes its NVFP4 experts to BF16; it requires a separate Python environment with PyTorch, Transformers, and safetensors. Do not enable value tracing during CUDA Graph capture.

## GEMV benchmark and Evolith

`kernel_bench` checks the BF16 GEMV output against a CPU reference and records alternating CUDA-event timings for a controlled block size. Allowed values are 64, 128, 256, and 512 threads; the runtime default is 128 unless a full-model comparison demonstrates a better incumbent.

```bash
target/release/kernel_bench --threads 256 --baseline-threads 128 --rounds 19 --iterations 128
QWEN_GEMV_THREADS=256 target/release/run \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --gpus 0,1,2,3,4,5,6,7 --prompt Hello --decode 64 --max-seq 80 --graph
```

The [Evolith](https://github.com/EeroEternal/evolith) `qwen-ultrafast` command can search this bounded space remotely. It requires microbenchmark acceptance, exact 64-token equality with the current incumbent, and a passing full-model AB/BA timing gate before reporting a successor. It does not change the runtime default automatically.

## Load the model

```bash
cargo run --release --bin run -- \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --gpus 0,1,2,3,4,5,6,7 \
  --load-only
```

The loader accounts for resident weight bytes per rank but does not upload the large PLE table. At eight ranks, stored resident weights are 17.42 GiB per GPU before execution state, NCCL, and other processes. Model loading performs a coordinated failure check so one failed rank does not leave the other ranks inside the expert reduction barrier.

## Generate

```bash
export QWEN_NCCL_LIBRARY=/path/to/libnccl.so.2
cargo run --release --bin run -- \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --gpus 0,1,2,3,4,5,6,7 \
  --prompt "Hello" \
  --decode 3 \
  --max-seq 2048 \
  --graph
```

NCCL 2.30's `ncclFloat32=7` is used for in-place FP32 expert reduction. `NCCL_ALGO=TREE` is the default unless explicitly overridden. If NCCL is not selected, a host-reduction path is available. The initial mode uses single-token prefill and is intended for numerical bring-up, not performance comparisons.

## SGLang comparison

The reference uses SGLang 0.5.21, PyTorch 2.13, TP8/EP8, `flashinfer_cutlass`, RING, deterministic inference, and decode graph. Deterministic mode is required: the default SM120/W4A4/EP path produced different tokens across otherwise identical requests. The checkpoint's existing `quantization_config` must also be written to `hf_quant_config.json` because this checkpoint does not include that optional SGLang metadata file.

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("models/Qwen3.8-Flash-Next-NVFP4")
config = json.loads((root / "config.json").read_text())
(root / "hf_quant_config.json").write_text(
    json.dumps(config["quantization_config"], indent=2) + "\n"
)
PY

NCCL_ALGO=RING .venv/bin/sglang serve \
  --model-path models/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 8 --ep-size 8 \
  --moe-runner-backend flashinfer_cutlass \
  --enable-deterministic-inference \
  --json-model-override-args '{"language_model_only":true}' \
  --context-length 80 \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-backend-prefill=disabled \
  --port 8001

.venv/bin/python python/sglang_generate.py \
  --url http://127.0.0.1:8001 --tokens 64 --runs 3
```

`python/sglang_generate.py` sends `input_ids=[9419]` with `temperature=0`, `top_k=1`, and `top_p=1`. Port 8000 on this machine belongs to another service; the baseline uses 8001.

## Layout

- `crates/cuda`: CUDA Driver API RAII, buffers, streams, events, modules, and graphs.
- `crates/kernels`: SM120 cubin build and CUDA C++ kernels.
- `crates/model`: nested Qwen configuration and memory-mapped safetensors access.
- `crates/engine`: typed launches, checkpoint mapping, model runtime, CPU references, and CLIs.

Qwen architecture references: [official model overview](https://github.com/QwenLM/Qwen3.8-Flash-Next) and [checkpoint configuration](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json).
