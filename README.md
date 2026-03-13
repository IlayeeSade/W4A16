# W4A16 LLaMA Benchmark Repo

This repository compares a regular causal language model against this repo's W4A16 variants on:

1. WikiText-2 perplexity
2. Single-layer forward speed
3. Full-model one-token decode speed

It also includes:

- a CUDA extension source for a custom W4A16 GEMV kernel
- a forward benchmark that can optionally compare against an LMDeploy AWQ model for speed only

## Variant Names

- `Regular BF16`: the baseline Hugging Face model with no quantization.
- `Repo CUDA Kernel W4A16`: this repo's quantized path using the local CUDA extension in [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu).
- `Repo Direct-Input CUDA W4A16`: the same kernel, but wrapped by [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu) so Python `forward()` passes the original activation tensor directly.
- `LMDeploy AWQ`: LMDeploy's own AWQ implementation, benchmarked only for speed when you provide an LMDeploy-compatible model path.

## What Is Actually In This Repo

```text
.
├── AGENTS.md
├── evaluation.py
├── forward_pass_benchmark.py
├── initial_script.py
├── quantization.py
├── sanity_checks.py
├── w4a16_cuda.cu
├── perplexity_results.json
├── perplexity_results.csv
└── perplexity_run.log
```

## Current Implementation

### Quantization path

- `quantization.py` implements naive per-group asymmetric 4-bit weight quantization.
- `QuantizedLinear4bit` stores packed 4-bit weights and dequantizes them on the fly in Python during `forward`.
- `CudaKernelQuantizedLinear4bit` uses the local CUDA extension in `w4a16_cuda.cu`.
- `CudaDirectQuantizedLinear4bit` uses `w4a16_cuda_direct.cu` to move activation flatten/transpose work out of Python `forward()` while keeping the kernel body unchanged.
- `quantize_model_layers(...)` deep-copies a model and replaces every `nn.Linear`.

### Perplexity path

- `initial_script.py` loads `meta-llama/Meta-Llama-3.1-8B`
- quantizes all `nn.Linear` layers with group size `64`
- evaluates both regular and quantized models on the WikiText-2 test split

### Benchmark path

`forward_pass_benchmark.py` benchmarks:

- `Regular BF16`: baseline `nn.Linear` / baseline full model
- `Repo CUDA Kernel W4A16`: this repo's local CUDA kernel path
- `Repo Direct-Input CUDA W4A16`: same kernel, but with activation layout preparation moved into the extension wrapper
- optional `LMDeploy AWQ`: an LMDeploy AWQ model path, speed only, not perplexity

It now also reports:

- `layer_breakdown_ms`: separate timing for input preparation, raw CUDA extension time, output post-processing, and total single-layer latency
- optional `profiled_model_layer_ms`: one model-layer breakdown using the exact hidden-state shape observed during decode
- environment metadata such as GPU name, CUDA capability, Torch version, timestamp, and kernel accumulation dtype

## Portable Benchmarking

Use [portable_benchmark.py](/workspace/W4A16/portable_benchmark.py) when you want one script that is easy to copy to another CUDA machine and run there.

Key points:

- synthetic mode benchmarks dense baseline vs repo kernel vs direct-input repo kernel
- full-model mode benchmarks a regular Hugging Face model against the repo-kernel quantized model
- dense baselines use BF16 when available and fall back to FP16 otherwise
- the repo CUDA kernel still uses BF16 tensors internally because that is the current extension contract

Examples:

```bash
python /workspace/W4A16/portable_benchmark.py \
  --mode synthetic \
  --results-json portable_results.json
```

```bash
python /workspace/W4A16/portable_benchmark.py \
  --model-checkpoint meta-llama/Meta-Llama-3.1-8B \
  --hf-token "$HF_TOKEN" \
  --results-json portable_model_results.json
```

Why this helps on an RTX 3070:

- dense baselines usually want FP16 there unless BF16 is explicitly reported as supported
- the script records the chosen dense dtype and the repo-kernel dtype in JSON so you can tell exactly what ran

## Known Runtime Constraint

The Llama 8B checkpoint does work on this machine, but it does **not** fit if Hugging Face caches into `/workspace/.hf_home` on the small overlay filesystem.

Use:

```bash
mkdir -p /dev/shm/hf_home /dev/shm/tmp
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/initial_script.py
```

Use the same cache redirection for the benchmark:

```bash
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/forward_pass_benchmark.py --hf-token "$HF_TOKEN" --enable-cuda-kernel --enable-direct-cuda-kernel
```

## Perplexity Result In This Repo

From [perplexity_results.json](/workspace/W4A16/perplexity_results.json):

- Model: `meta-llama/Meta-Llama-3.1-8B`
- Dataset: `wikitext-2-raw-v1:test`
- Regular perplexity: `13.0328`
- Quantized perplexity: `15.3972`
- Delta: `+2.3644`
- Regular model size: `15316.51 MB`
- Quantized model size: `5028.32 MB`

## Benchmark Result In This Repo

From the latest smoke run in [benchmark_run.log](/workspace/W4A16/benchmark_run.log) and [benchmark_results.json](/workspace/W4A16/benchmark_results.json):

![Benchmark Plot](./benchmark_plot.png)

The plot is meant to answer one question quickly:

- `Regular BF16` is the baseline to beat.
- `Repo CUDA Kernel W4A16` is the original repo wrapper.
- `Repo Direct-Input CUDA W4A16` removes Python-side reshape/transpose from `forward()`, but keeps the same kernel body.
- `LMDeploy AWQ`, when provided, is an external reference point for speed only.

### Single Linear Layer

- `Regular BF16`: `0.058 ms`
- `Repo CUDA Kernel W4A16`: `0.272 ms`
- `Repo Direct-Input CUDA W4A16`: `0.265 ms`

### Full Model, One Decode Token

- `Regular BF16`: `38.477 ms`
- `Repo CUDA Kernel W4A16`: `108.699 ms`
- `Repo Direct-Input CUDA W4A16`: `108.526 ms`

These numbers show the intended comparison clearly:

- moving the Python reshape/transpose into a new CUDA host wrapper does not materially change the result
- both repo-local CUDA kernel paths are still slower than the regular BF16 baseline in the current end-to-end benchmark
- the important question is now isolated kernel speed versus full-model integration cost and BF16 library advantages
- [explanation.md](/workspace/W4A16/explanation.md) explains the gap and records the isolated benchmark result

## LMDeploy Comparison

`forward_pass_benchmark.py` supports an optional LMDeploy speed-only comparison:

```bash
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/forward_pass_benchmark.py \
  --hf-token "$HF_TOKEN" \
  --enable-cuda-kernel \
  --enable-direct-cuda-kernel \
  --lmdeploy-model-path /path/to/awq-model \
  --lmdeploy-backend pytorch
```

Notes:

- this comparison is **speed only**
- it does **not** participate in the perplexity script
- it expects an already prepared LMDeploy-compatible AWQ model path
- the benchmark measures end-to-end generation latency for `max_new_tokens=1`, not raw Hugging Face `forward(...)`

Current status on this machine:

- an LMDeploy AWQ artifact was generated successfully at `/workspace/W4A16/lmdeploy_awq_llama31_8b`
- benchmarking it is currently blocked by the local LMDeploy runtime, not by the model artifact
- the failing path is LMDeploy's Triton/Inductor startup on this CUDA 13 / Blackwell environment, so there is no valid LMDeploy latency number in the repo yet

## Files

- [initial_script.py](/workspace/W4A16/initial_script.py): perplexity entrypoint
- [evaluation.py](/workspace/W4A16/evaluation.py): dataset prep and perplexity evaluation
- [quantization.py](/workspace/W4A16/quantization.py): quantization helpers and quantized linear modules
- [sanity_checks.py](/workspace/W4A16/sanity_checks.py): forward-output comparison helper
- [forward_pass_benchmark.py](/workspace/W4A16/forward_pass_benchmark.py): speed benchmark entrypoint
- [kernel_only_benchmark.py](/workspace/W4A16/kernel_only_benchmark.py): isolated dense-BF16 vs kernel-only GEMV benchmark
- [previous_benchmarking.py](/workspace/W4A16/previous_benchmarking.py): older isolated benchmark script used as a reference point for kernel-vs-BF16 intuition
- [explanation.md](/workspace/W4A16/explanation.md): why the isolated kernel result can look faster while the repo full-model benchmark stays slower
- [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu): CUDA extension source
- [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu): same kernel structure with a different host wrapper so Python `forward()` passes activations directly

## What This README Does Not Claim

This repo does **not** currently contain:

- a packaged build system such as `setup.py`
- a `requirements.txt`
- a `benchmarks/`, `tests/`, `eval/`, or `quantize/` directory layout
- validated production W4A16 throughput improvements end to end
- an integrated LMDeploy perplexity path
