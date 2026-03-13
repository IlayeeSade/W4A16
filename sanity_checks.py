import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


@torch.no_grad()
def test_model_vs_quantized_model_forward(
    model_checkpoint: str,
    quantized_model: nn.Module,
    hf_token: str,
    batch_size: int = 1,
    seq_len: int = 16,
    atol: float = 2e-1,
    rtol: float = 2e-1,
):
    device = torch.device("cuda")

    print(f"Loading original model from {model_checkpoint} to GPU ...")
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, token=hf_token)
    model.to(device).eval()

    vocab_size = getattr(model.config, "vocab_size", 128256)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    print("Running original model ...")
    ref_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).logits.float().cpu()

    del model
    torch.cuda.empty_cache()

    print("Moving quantized model to GPU ...")
    quantized_model.to(device).eval()
    q_logits = quantized_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    ).logits.float().cpu()

    diff = (ref_logits - q_logits).abs()
    print("=== Logits comparison ===")
    print("max abs diff    :", diff.max().item())
    print("mean abs diff   :", diff.mean().item())
    print(
        "relative L2 err :",
        (torch.norm(ref_logits - q_logits) / (torch.norm(ref_logits) + 1e-12)).item(),
    )
    print("allclose        :", torch.allclose(ref_logits, q_logits, atol=atol, rtol=rtol))

    quantized_model.cpu()
    torch.cuda.empty_cache()
