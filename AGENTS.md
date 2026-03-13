# W4A16 Project Guide

## Purpose
This project compares language-model perplexity before and after 4-bit weight quantization.
The current workflow:

1. Load a causal LM checkpoint.
2. Replace `nn.Linear` layers with `QuantizedLinear4bit`.
3. Evaluate perplexity for the regular model on WikiText-2.
4. Evaluate perplexity for the quantized model on WikiText-2.
5. Print a side-by-side summary.

## Repository Layout
- `initial_script.py`: main entrypoint for loading the model, quantizing it, and running both perplexity evaluations.
- `quantization.py`: weight packing, quantization, dequantization, and `QuantizedLinear4bit`.
- `evaluation.py`: WikiText-2 preparation and perplexity calculation.
- `sanity_checks.py`: forward-pass comparison helper for regular vs quantized logits.
- `w4a16_cuda.cu`: CUDA extension source for a custom W4A16 forward path.

## Known Runtime Constraint
The default Hugging Face cache path in this environment points into `/workspace`, which sits on a small overlay filesystem.
`meta-llama/Meta-Llama-3.1-8B` does load correctly on this machine, but it fails if the cache lives on `/workspace` because the model shards exhaust the overlay.

Use:

```bash
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/initial_script.py
```

`/dev/shm` provides enough temporary space for the active model download/load path in this environment.

## Operational Notes
- The current script uses `meta-llama/Meta-Llama-3.1-8B`.
- The current script expects a valid Hugging Face token for gated model access.
- GPU inference is used when CUDA is available.
- CPU RAM on this machine is sufficient to hold both the original and quantized model copies.
- The quantized forward path currently dequantizes on the fly, so it is expected to be slower than the regular model even if memory use is reduced.

## Status Meanings
- `Loading tokenizer/model`: Hugging Face checkpoint access and weight loading.
- `Preparing WikiText-2 dataset`: dataset download/tokenization/chunking.
- `Quantizing model`: replacing `nn.Linear` layers with quantized modules.
- `Calculating perplexity`: evaluating average cross-entropy over the WikiText-2 test split.
- `Perplexity Comparison`: final result summary for regular vs quantized models.

## Recommended Run Procedure
1. Clear partial failed checkpoint downloads if `/workspace/.hf_home` filled up during a prior run.
2. Run the script with `HF_HOME=/dev/shm/hf_home` and `TMPDIR=/dev/shm/tmp`.
3. Capture the console output to a log if you want a persistent artifact:

```bash
mkdir -p /dev/shm/hf_home /dev/shm/tmp
HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp python /workspace/W4A16/initial_script.py | tee /workspace/W4A16/perplexity_run.log
```
