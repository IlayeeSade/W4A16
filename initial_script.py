"""
Perplexity evaluation entrypoint.

Loads Meta-Llama-3.1-8B, quantizes its linear layers to 4-bit (W4A16),
and evaluates perplexity on the WikiText-2 test split.

Requirements:
    pip install torch transformers datasets tqdm accelerate
"""

import gc
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation import calculate_perplexity, prepare_lm_datasets
from quantization import (
    count_unquantized_linear_layers,
    get_model_size_mb,
    quantize_model_layers,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_CHECKPOINT = "meta-llama/Meta-Llama-3.1-8B"
GROUP_SIZE = 64
BLOCK_SIZE = 1024
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set in the environment to access the gated model.")

    # 1. Load tokenizer & model
    print(f"Loading tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, token=HF_TOKEN)

    print(f"Loading model: {MODEL_CHECKPOINT} (bfloat16, CPU)")
    model_on_cpu = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT,
        dtype=torch.bfloat16,
        token=HF_TOKEN,
        device_map="cpu",
    )
    print("Model loaded.")

    # 2. Prepare dataset
    print("Preparing WikiText-2 dataset ...")
    lm_datasets = prepare_lm_datasets(tokenizer, BLOCK_SIZE)
    print(lm_datasets)

    # 3. Size of original model
    print(f"Original model size: {get_model_size_mb(model_on_cpu):.2f} MB")

    # 4. Quantize
    print("Quantizing model ...")
    regular_model, quantized_model = quantize_model_layers(model_on_cpu, GROUP_SIZE)
    print(f"Quantized model size: {get_model_size_mb(quantized_model):.2f} MB")

    # 5. Structural check
    linear_found, quantized_count = count_unquantized_linear_layers(quantized_model)
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  [WARNING] Unquantized nn.Linear: {name}")
    if not linear_found:
        print(f"All nn.Linear layers replaced. Total QuantizedLinear4bit: {quantized_count}")

    # 6. Evaluate regular model perplexity
    print(f"\nMoving regular model to {DEVICE} ...")
    regular_model.to(DEVICE)
    regular_model_perplexity = calculate_perplexity(regular_model, lm_datasets["test"], batch_size=1)
    print(f"Regular model perplexity: {regular_model_perplexity:.4f}")
    regular_model.cpu()
    del regular_model
    torch.cuda.empty_cache()
    gc.collect()

    # 7. Evaluate quantized model perplexity
    print(f"\nMoving quantized model to {DEVICE} ...")
    quantized_model.to(DEVICE)
    quantized_model_perplexity = calculate_perplexity(quantized_model, lm_datasets["test"], batch_size=1)
    print(f"Quantized model perplexity: {quantized_model_perplexity:.4f}")
    quantized_model.cpu()
    del quantized_model
    torch.cuda.empty_cache()
    gc.collect()

    # 8. Summary
    print("\n=== Perplexity Comparison ===")
    print(f"  Regular model   : {regular_model_perplexity:.4f}")
    print(f"  Quantized model : {quantized_model_perplexity:.4f}")


if __name__ == "__main__":
    main()
