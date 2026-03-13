# W4A16 Project Guide

## Purpose
This repo studies one question: how a custom W4A16 quantized path compares with the regular BF16 Hugging Face path for `meta-llama/Meta-Llama-3.1-8B`.

The project has three active tracks:

1. perplexity on WikiText-2
2. forward-speed benchmarking
3. kernel-integration analysis

## Main Variants
- `Regular BF16`: unquantized Hugging Face baseline.
- `QuantizedLinear4bit`: Python dequantization path, used for perplexity experiments.
- `Repo CUDA Kernel W4A16`: repo kernel wrapper using [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu).
- `Repo Direct-Input CUDA W4A16`: same kernel body, different host wrapper in [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu).
- `LMDeploy AWQ`: external speed-only comparison target.

## Core Files
- [initial_script.py](/workspace/W4A16/initial_script.py): main perplexity entrypoint.
- [evaluation.py](/workspace/W4A16/evaluation.py): WikiText-2 loading and perplexity calculation.
- [quantization.py](/workspace/W4A16/quantization.py): packing, quantization, dequantization, and quantized linear modules.
- [sanity_checks.py](/workspace/W4A16/sanity_checks.py): forward correctness check.
- [forward_pass_benchmark.py](/workspace/W4A16/forward_pass_benchmark.py): single-layer and full-model latency benchmark.
- [kernel_only_benchmark.py](/workspace/W4A16/kernel_only_benchmark.py): isolated dense-BF16 vs kernel-only comparison.
- [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu): original CUDA kernel wrapper.
- [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu): same kernel structure with direct-input host wrapper.
- [previous_benchmarking.py](/workspace/W4A16/previous_benchmarking.py): earlier synthetic reference benchmark.
- [README.md](/workspace/W4A16/README.md): user-facing repo summary.
- [explanation.md](/workspace/W4A16/explanation.md): why the kernel still loses to BF16 here.
- [story_explanation.md](/workspace/W4A16/story_explanation.md): full session narrative and experiment history.

## Current Known Results
- Perplexity:
  regular `13.0328`
  quantized `15.3972`
- Full-model decode latency:
  regular BF16 `38.477 ms`
  repo kernel `108.699 ms`
  repo direct-input kernel `108.526 ms`
- Kernel-only benchmark:
  dense BF16 still beats both repo kernel variants across tested sizes.

## Runtime Constraints
- Use `/dev/shm` for Hugging Face cache and temp files on this machine.
- Recommended env:
  `HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp`
- The Llama 8B checkpoint is gated and needs a valid `HF_TOKEN`.
- LMDeploy AWQ artifact generation works locally, but LMDeploy runtime benchmarking is currently blocked by the environment on this CUDA 13 / Blackwell stack.

## Recommended Commands
Perplexity:
```bash
mkdir -p /dev/shm/hf_home /dev/shm/tmp
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/initial_script.py
```

Repo benchmark:
```bash
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/forward_pass_benchmark.py \
  --hf-token "$HF_TOKEN" \
  --enable-cuda-kernel \
  --enable-direct-cuda-kernel
```

Kernel-only benchmark:
```bash
python /workspace/W4A16/kernel_only_benchmark.py
```

## Working Rules For Future Changes
- Keep the original kernel in [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu) as the baseline unless the task explicitly says to change it.
- Put kernel experiments in separate CUDA files when possible.
- Treat LMDeploy as speed-only unless a task explicitly asks for something else.
- Keep commits small and milestone-based.
