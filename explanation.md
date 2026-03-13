# Why The Kernel Is Still Slower Than BF16 In This Repo

## Short Answer

The new direct-input wrapper proves that Python-side reshape/transpose is not the main bottleneck.

The measured results are still:

- full model decode: `Regular BF16 38.477 ms`
- full model decode: `Repo CUDA Kernel W4A16 108.699 ms`
- full model decode: `Repo Direct-Input CUDA W4A16 108.526 ms`
- isolated GEMV: `dense BF16 0.1370 ms` vs `repo kernel 0.4756 ms` vs `direct-input kernel 0.4819 ms` at `OF=4096, IF=8192`

So the current kernel path is slower than BF16 even in the isolated benchmark on this machine.

## What Changed

I added [w4a16_cuda_direct.cu](/workspace/W4A16/w4a16_cuda_direct.cu) and [CudaDirectQuantizedLinear4bit in quantization.py](/workspace/W4A16/quantization.py).

That change keeps the same kernel body and only changes the host wrapper contract:

- old path: Python `forward()` flattened and transposed activations before calling the extension
- new path: Python `forward()` passes the original activation tensor directly
- new host wrapper: flattens/transposes internally and restores the output shape before returning

This was the cleanest way to test "is Python-side shape handling the reason the repo path is slow?" without changing the kernel itself.

The answer is no, or at least not in a way that matters here:

- single layer: `0.272 ms` -> `0.265 ms`
- full model: `108.699 ms` -> `108.526 ms`

That is a tiny change, not a regime change.

## Why `previous_benchmarking.py` Looked More Promising

[previous_benchmarking.py](/workspace/W4A16/previous_benchmarking.py) is still useful, but it is not the same experiment as the repo benchmark.

It is closer to a best-case synthetic setup:

- random tensors
- direct kernel invocation
- one GEMV-like operation at a time
- no Hugging Face decode stack
- no repeated module traversal across the full transformer

That means it can support the intuition that "the kernel should be competitive" without proving that the current repo integration beats the BF16 baseline end to end.

## What The Isolated Benchmark Says Now

[kernel_only_benchmark.py](/workspace/W4A16/kernel_only_benchmark.py) measures the kernel more directly against a dense BF16 baseline using the same shapes as the repo experiments.

Current results:

- `OF=4096`: dense `0.1370 ms`, repo kernel `0.4756 ms`, direct-input kernel `0.4819 ms`
- `OF=8192`: dense `0.2454 ms`, repo kernel `0.9305 ms`, direct-input kernel `0.9363 ms`
- `OF=16384`: dense `0.4618 ms`, repo kernel `1.7281 ms`, direct-input kernel `1.7365 ms`
- `OF=32768`: dense `0.8984 ms`, repo kernel `3.4278 ms`, direct-input kernel `3.4315 ms`

So in this environment I could not verify the claim that the current kernel is faster than dense BF16, even in the isolated microbenchmark.

## Why BF16 Still Wins

The dense BF16 baseline is not just "plain matmul". It is using mature library paths that the custom kernel does not get for free.

I checked the active BLAS backend:

- `torch.backends.cuda.preferred_blas_library() -> _BlasBackend.Cublas`

I also profiled BF16 `torch.matmul(W, X)` on CUDA. The profiler shows:

- `aten::matmul`
- `aten::mm`
- an internal `gemvx`-style backend under the CUDA work

That means the dense BF16 baseline is benefiting from PyTorch + cuBLAS-backed kernels that are already tuned for this GPU and software stack.

In practice that means:

- the BF16 baseline has better kernel selection and dispatch
- the BF16 path is likely getting hardware/library optimizations that our custom kernel wrapper does not match yet
- beating BF16 requires more than removing a Python reshape

## What This Means For The Repo

The current repo gap is not explained by one obvious wrapper bug anymore.

The evidence now points to a more fundamental issue:

- the kernel itself is not yet faster than the dense BF16 baseline on this machine
- the full-model integration naturally stays slower because it stacks that slower kernel across many layers

So the next useful experiments should stay outside [w4a16_cuda.cu](/workspace/W4A16/w4a16_cuda.cu) and preserve the current kernel as the baseline:

1. try kernel changes in a separate experimental CUDA file
2. measure one transformer block, not only kernel-only and full-model extremes
3. compare BF16 and FP16 under exactly the same isolated conditions
4. inspect whether dequantization math, memory access pattern, or launch geometry is the main limiter

## LMDeploy Status

I also generated an LMDeploy AWQ artifact successfully at:

- `/workspace/W4A16/lmdeploy_awq_llama31_8b`

But I could not produce a valid LMDeploy speed number on this machine.

What failed:

- first LMDeploy was missing `xgrammar`
- after installing it, the LMDeploy runtime failed in Triton/Inductor startup
- the error occurs on this CUDA 13 / Blackwell environment before a stable generation benchmark can run

So the repo now has:

- a valid local LMDeploy AWQ model artifact
- a reproduced LMDeploy runtime failure
- no trustworthy LMDeploy latency number yet
