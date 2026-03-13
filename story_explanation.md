# Session Story And Project Explanation

## What This Project Is

This repo is a focused benchmark and evaluation project around one model family and one quantization idea.

The target model is:

- `meta-llama/Meta-Llama-3.1-8B`

The core question is:

- how the regular BF16 model compares with this repo's W4A16 quantized implementations

That question is tested through three pipelines:

1. a perplexity pipeline on WikiText-2
2. a forward-pass latency benchmark
3. an isolated kernel-vs-BF16 microbenchmark

There is also one external comparison track:

- LMDeploy AWQ, for speed only

## What The Repo Includes

### Perplexity pipeline

- [initial_script.py](/workspace/W4A16/initial_script.py)
- [evaluation.py](/workspace/W4A16/evaluation.py)
- [quantization.py](/workspace/W4A16/quantization.py)
- [sanity_checks.py](/workspace/W4A16/sanity_checks.py)

This pipeline:

1. loads the Llama 8B model
2. quantizes all `nn.Linear` layers
3. runs a forward sanity check
4. evaluates perplexity on WikiText-2 for the regular model
5. evaluates perplexity on WikiText-2 for the quantized model
6. writes results and logs

### Forward benchmark pipeline

- [forward_pass_benchmark.py](/workspace/W4A16/forward_pass_benchmark.py)
- [quantization.py](/workspace/W4A16/quantization.py)
- [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu)
- [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu)

This pipeline measures:

- single linear layer latency
- one-token full-model decode latency

for these variants:

- `Regular BF16`
- `Repo CUDA Kernel W4A16`
- `Repo Direct-Input CUDA W4A16`
- optional `LMDeploy AWQ`

### Kernel-only benchmark pipeline

- [kernel_only_benchmark.py](/workspace/W4A16/kernel_only_benchmark.py)
- [quantization.py](/workspace/W4A16/quantization.py)
- [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu)
- [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu)

This pipeline strips the problem down to a dense BF16 GEMV-style baseline versus direct extension calls to the repo kernels.

### Analysis and docs

- [README.md](/workspace/W4A16/README.md)
- [explanation.md](/workspace/W4A16/explanation.md)
- [lmdeploy_status.json](/workspace/W4A16/lmdeploy_status.json)

These files explain what the repo actually does, what was tested, what worked, and what failed.

## What I Changed Across The Session

### 1. Split the original monolithic script

The original single file mixed:

- quantization helpers
- perplexity evaluation
- forward sanity checks

I separated those concerns into:

- [quantization.py](/workspace/W4A16/quantization.py)
- [evaluation.py](/workspace/W4A16/evaluation.py)
- [sanity_checks.py](/workspace/W4A16/sanity_checks.py)

and kept [initial_script.py](/workspace/W4A16/initial_script.py) as the entrypoint.

### 2. Made the Llama 8B run actually work on this machine

The model was not fundamentally broken. The real issue was storage.

The failing path was the default Hugging Face cache landing on the small `/workspace` overlay filesystem. The fix was to move cache and temp directories to shared memory:

```bash
mkdir -p /dev/shm/hf_home /dev/shm/tmp
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/initial_script.py
```

That allowed the 8B checkpoint to load and run successfully.

### 3. Ran and exported perplexity results

I ran the regular and quantized model on WikiText-2 and wrote:

- [perplexity_run.log](/workspace/W4A16/perplexity_run.log)
- [perplexity_results.json](/workspace/W4A16/perplexity_results.json)
- [perplexity_results.csv](/workspace/W4A16/perplexity_results.csv)

Measured result:

- regular perplexity: `13.0328`
- quantized perplexity: `15.3972`
- delta: `+2.3644`

I also recorded model-size reduction:

- regular size: `15316.51 MB`
- quantized size: `5028.32 MB`

### 4. Added and cleaned up benchmark naming

The benchmark naming was ambiguous, so I made it explicit:

- `Regular BF16`
- `Repo CUDA Kernel W4A16`
- `Repo Direct-Input CUDA W4A16`
- `LMDeploy AWQ`

I also removed the no-longer-useful local "python W4A16" benchmark from the active benchmark comparison.

### 5. Tested the repo kernel path and found it slower than BF16

I benchmarked the repo CUDA kernel against the regular BF16 baseline.

Single-layer result:

- `Regular BF16`: `0.058 ms`
- `Repo CUDA Kernel W4A16`: `0.272 ms`

Full-model decode result:

- `Regular BF16`: `38.477 ms`
- `Repo CUDA Kernel W4A16`: `108.699 ms`

So the repo kernel path was clearly slower than the regular BF16 path.

### 6. Added a new CUDA wrapper experiment without changing the kernel body

You asked for a new CUDA file that keeps the same kernel structure but avoids Python-side reshape/manipulation in quantized `forward()`.

I implemented that as:

- [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu)

and connected it through:

- `CudaDirectQuantizedLinear4bit` in [quantization.py](/workspace/W4A16/quantization.py)

Important detail:

- I did not change the underlying kernel body
- I changed only the host-wrapper contract
- the new wrapper accepts the original activation shape and does the flatten/transpose internally

That means this was a controlled test of wrapper overhead, not a new kernel algorithm.

### 7. Measured the direct-input wrapper

Then I reran the benchmark with the new direct-input wrapper.

Single-layer result:

- `Regular BF16`: `0.058 ms`
- `Repo CUDA Kernel W4A16`: `0.272 ms`
- `Repo Direct-Input CUDA W4A16`: `0.265 ms`

Full-model result:

- `Regular BF16`: `38.477 ms`
- `Repo CUDA Kernel W4A16`: `108.699 ms`
- `Repo Direct-Input CUDA W4A16`: `108.526 ms`

Conclusion:

- moving reshape/transpose out of Python did not materially improve the result

### 8. Added an isolated kernel-only benchmark

To test whether the repo kernel at least wins in a cleaner environment, I added:

- [kernel_only_benchmark.py](/workspace/W4A16/kernel_only_benchmark.py)

This directly compares:

- dense BF16 `torch.matmul(...) + bias`
- repo kernel extension
- direct-input repo kernel extension

Measured results:

- `OF=4096`: dense `0.1370 ms`, repo `0.4756 ms`, direct `0.4819 ms`
- `OF=8192`: dense `0.2454 ms`, repo `0.9305 ms`, direct `0.9363 ms`
- `OF=16384`: dense `0.4618 ms`, repo `1.7281 ms`, direct `1.7365 ms`
- `OF=32768`: dense `0.8984 ms`, repo `3.4278 ms`, direct `3.4315 ms`

Conclusion:

- on this machine, the current kernel still loses to dense BF16 even in the isolated microbenchmark

### 9. Checked why BF16 is so strong

I explicitly checked whether the BF16 baseline was getting better library support than the isolated custom path.

I verified:

- `torch.backends.cuda.preferred_blas_library() -> _BlasBackend.Cublas`

I also profiled BF16 `torch.matmul` and saw:

- `aten::matmul`
- `aten::mm`
- an internal `gemvx`-style backend under CUDA

That matters because it means the dense BF16 baseline is using mature PyTorch/cuBLAS-backed paths that are already heavily optimized for this GPU and software stack.

So the repo kernel is not just losing to "plain dense math". It is losing to a very optimized vendor/library path.

### 10. Tried to bring LMDeploy into the comparison

You asked for an LMDeploy AWQ speed comparison.

I successfully generated a local AWQ artifact:

- `/workspace/W4A16/lmdeploy_awq_llama31_8b`

using LMDeploy's AWQ tooling on the cached local model snapshot.

But benchmarking that artifact failed on this environment.

The failure progression was:

1. LMDeploy initially failed because `xgrammar` was missing
2. after installing that, LMDeploy still failed in Triton/Inductor startup
3. the failure happens on this CUDA 13 / Blackwell environment before a stable latency run completes

So:

- artifact generation succeeded
- runtime benchmarking did not
- there is no trustworthy LMDeploy latency number in the repo yet

That status is saved in [lmdeploy_status.json](/workspace/W4A16/lmdeploy_status.json).

### 11. Updated documentation to match reality

I updated or added:

- [README.md](/workspace/W4A16/README.md)
- [explanation.md](/workspace/W4A16/explanation.md)
- [AGENTS.md](/workspace/W4A16/AGENTS.md)
- this file, [story_explanation.md](/workspace/W4A16/story_explanation.md)

The goal was to make the repo self-describing and aligned with actual code and measured results, not earlier assumptions.

### 12. Maintained milestone commits and pushes

During the session I created and pushed milestone commits as the repo changed shape.

The latest pushed commit before this new documentation update was:

- `f259c67` `Add direct-input kernel wrapper analysis`

## Exact Run Commands Used

### Perplexity run

```bash
mkdir -p /dev/shm/hf_home /dev/shm/tmp
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/initial_script.py
```

### Repo forward benchmark

```bash
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/forward_pass_benchmark.py \
  --hf-token "$HF_TOKEN" \
  --enable-cuda-kernel \
  --enable-direct-cuda-kernel \
  --results-json /workspace/W4A16/benchmark_results.json \
  --plot-path /workspace/W4A16/benchmark_plot.png
```

### Kernel-only benchmark

```bash
python /workspace/W4A16/kernel_only_benchmark.py
```

### BF16 backend check

I ran checks equivalent to:

```bash
python - <<'PY'
import torch
print(torch.backends.cuda.preferred_blas_library())
PY
```

and a profiler run around BF16 `torch.matmul(...)`.

### LMDeploy AWQ generation

I used LMDeploy AWQ tooling against the cached local snapshot in `/dev/shm/hf_home/...`.

### LMDeploy benchmark attempts

I tried both:

- LMDeploy PyTorch backend
- LMDeploy TurboMind backend

Both failed before producing a stable benchmark number in this environment.

## What The Current Project State Means

Right now the repo supports a clear technical story:

- perplexity evaluation works
- regular vs quantized perplexity is measured and exported
- repo forward benchmarking works
- the current repo kernel is slower than regular BF16
- the direct-input wrapper experiment shows that Python reshape/transpose is not the main reason
- the isolated kernel benchmark shows the kernel still loses to dense BF16 here
- BF16 is benefiting from strong library support
- LMDeploy artifact generation works, but runtime benchmarking is still environment-blocked

## What I Would Consider The Next Correct Steps

If the goal is to make the repo kernel truly competitive, the next steps should be:

1. keep [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu) as the baseline
2. put kernel changes in a separate experimental CUDA file
3. benchmark one transformer block, not only full model and kernel-only extremes
4. compare exactly matched BF16 and FP16 isolated runs
5. inspect kernel math and memory behavior instead of focusing only on Python wrapper cleanup

That is the state of the project at the end of this session.
