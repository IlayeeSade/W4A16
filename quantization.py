import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as load_torch_extension


def pack_rows_4(W_q: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit quantized weights row-wise.

    Args:
        W_q: [OF, IF], uint8 values in [0, 15]

    Returns:
        [OF // 4, IF], uint16  four rows packed into one
    """
    OF, IF = W_q.shape
    if OF % 4 != 0:
        raise ValueError("OF must be divisible by 4")

    W4 = W_q.contiguous().view(OF // 4, 4, IF).to(torch.int32)
    packed = (
        (W4[:, 0, :] << 0)
        | (W4[:, 1, :] << 4)
        | (W4[:, 2, :] << 8)
        | (W4[:, 3, :] << 12)
    )
    return packed.to(torch.uint16).contiguous()


def unpack_rows_4(W_packed: torch.Tensor, original_in_features: int) -> torch.Tensor:
    """
    Unpack 4-bit packed weights back to uint8.

    Args:
        W_packed: [OF//4, IF], packed uint16
        original_in_features: original IF dimension

    Returns:
        [OF, IF], uint8 values in [0, 15]
    """
    x = W_packed.to(torch.int32)

    r0 = (x >> 0) & 0xF
    r1 = (x >> 4) & 0xF
    r2 = (x >> 8) & 0xF
    r3 = (x >> 12) & 0xF

    W_q = torch.stack([r0, r1, r2, r3], dim=1)
    return W_q.view(-1, original_in_features).to(torch.uint8)


def quantize_weights(weight: torch.Tensor, group_size: int, debug: bool = False):
    """
    Quantize a weight matrix to 4-bit with per-group asymmetric quantization.

    Returns:
        W_packed : [OF//4, IF] packed uint16
        SZ       : [num_groups, 2*OF] interleaved scale / zero-point (bfloat16)
    """
    del debug

    OF, IF = weight.shape

    if IF % group_size != 0:
        raise ValueError("IF must be divisible by group_size")
    if OF % 4 != 0:
        raise ValueError("OF must be divisible by 4")

    num_groups = IF // group_size
    w = weight.float().view(OF, num_groups, group_size)

    min_vals = w.min(dim=-1, keepdim=True).values
    max_vals = w.max(dim=-1, keepdim=True).values

    S = (max_vals - min_vals) / 15.0
    zero_scale_mask = S == 0
    safe_S = torch.where(zero_scale_mask, torch.ones_like(S), S)

    Z_int = (-min_vals / safe_S).round().clamp(0, 15)
    Z_int = Z_int.masked_fill(zero_scale_mask, 0)

    W_q = ((w / safe_S) + Z_int).round().clamp(0, 15).to(torch.uint8)
    W_q = W_q.masked_fill(zero_scale_mask.expand_as(W_q), 0)

    W_q_2d = W_q.view(OF, IF).contiguous()
    W_packed = pack_rows_4(W_q_2d)

    S = S.squeeze(-1).to(torch.bfloat16).contiguous()
    Z = Z_int.squeeze(-1).to(torch.bfloat16).contiguous()

    SZ = torch.empty((num_groups, 2 * OF), dtype=torch.bfloat16, device=weight.device)
    SZ[:, 0::2] = S.transpose(0, 1)
    SZ[:, 1::2] = Z.transpose(0, 1)
    return W_packed, SZ.contiguous()


def dequantize_weights(
    W_packed: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    group_size: int,
    original_in_features: int,
) -> torch.Tensor:
    """
    Dequantize packed 4-bit weights back to float.

    Args:
        W_packed: [OF//4, IF]
        S, Z: [OF, num_groups] scale and integer zero-point
        group_size: quantization group size
        original_in_features: IF

    Returns:
        [OF, IF] float tensor
    """
    OF_div_4, _ = W_packed.shape
    OF = OF_div_4 * 4
    num_groups = original_in_features // group_size

    W_q = unpack_rows_4(W_packed, original_in_features)
    W_q_reshaped = W_q.view(OF, num_groups, group_size).float()
    S_reshaped = S.view(OF, num_groups, 1)
    Z_reshaped = Z.view(OF, num_groups, 1)

    W_dequant = (W_q_reshaped - Z_reshaped) * S_reshaped
    return W_dequant.view(OF, original_in_features)


class QuantizedLinear4bit(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores weights in 4-bit packed form
    and dequantizes on-the-fly during forward.

    Note: the original notebook used a CUDA kernel (w4a16_cuda_ext) for the
    forward pass. This version falls back to a pure-Python dequant + matmul
    so the code runs without building the extension.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.register_buffer("W_packed", None)
        self.register_buffer("SZ_packed", None)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16)) if bias else None

    @classmethod
    def from_linear(cls, linear_module: nn.Linear, group_size: int, debug: bool = False):
        W_packed, SZ_packed = quantize_weights(linear_module.weight.data, group_size, debug=debug)
        qmod = cls(
            linear_module.in_features,
            linear_module.out_features,
            group_size,
            linear_module.bias is not None,
        )
        qmod.W_packed = W_packed
        qmod.SZ_packed = SZ_packed
        if linear_module.bias is not None:
            qmod.bias.data.copy_(linear_module.bias.data.to(torch.bfloat16))
        return qmod

    def _dequant_weight(self) -> torch.Tensor:
        OF = self.W_packed.shape[0] * 4
        num_groups = self.in_features // self.group_size

        S = self.SZ_packed[:, 0::2].t().contiguous().view(OF, num_groups)
        Z = self.SZ_packed[:, 1::2].t().contiguous().view(OF, num_groups)
        return dequantize_weights(
            self.W_packed,
            S,
            Z,
            self.group_size,
            self.in_features,
        ).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequant_weight().to(x.device)
        return nn.functional.linear(x.to(torch.bfloat16), weight, self.bias)


def load_w4a16_cuda_extension(verbose: bool = False):
    source_path = Path(__file__).with_name("w4a16_cuda.cu")
    return load_torch_extension(
        name="w4a16_cuda_ext_v7",
        sources=[str(source_path)],
        verbose=verbose,
    )


class CudaKernelQuantizedLinear4bit(QuantizedLinear4bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        cuda_ext,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, group_size, bias=bias)
        self.cuda_ext = cuda_ext
        if bias:
            self.register_buffer("_zero_bias", None)
        else:
            self.register_buffer("_zero_bias", torch.zeros(out_features, dtype=torch.bfloat16))

    @classmethod
    def from_linear(cls, linear_module: nn.Linear, group_size: int, cuda_ext):
        W_packed, SZ_packed = quantize_weights(linear_module.weight.data, group_size)
        qmod = cls(
            linear_module.in_features,
            linear_module.out_features,
            group_size,
            cuda_ext=cuda_ext,
            bias=linear_module.bias is not None,
        )
        qmod.W_packed = W_packed
        qmod.SZ_packed = SZ_packed
        if linear_module.bias is not None:
            qmod.bias.data.copy_(linear_module.bias.data.to(torch.bfloat16))
        return qmod

    def _kernel_input(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.in_features).transpose(0, 1).contiguous().to(torch.bfloat16)

    def _kernel_bias(self) -> torch.Tensor:
        return self.bias if self.bias is not None else self._zero_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return super().forward(x)

        out = self.cuda_ext.forward(
            self.W_packed,
            self._kernel_bias(),
            self.SZ_packed,
            self._kernel_input(x),
            self.group_size,
        )
        return out.transpose(0, 1).contiguous().view(*x.shape[:-1], self.out_features)


def quantize_model_layers(model: nn.Module, group_size: int, linear_cls=QuantizedLinear4bit, **linear_kwargs):
    """
    Deep-copy the model and replace every nn.Linear with QuantizedLinear4bit.

    Returns:
        (original_model, quantized_model)
    """
    quantized_model = copy.deepcopy(model)

    def _quantize_inplace(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, linear_cls.from_linear(child, group_size, **linear_kwargs))
            else:
                _quantize_inplace(child)

    _quantize_inplace(quantized_model)
    return model, quantized_model


def get_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


def count_unquantized_linear_layers(model: nn.Module) -> tuple[bool, int]:
    linear_found = False
    quantized_count = 0

    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_found = True
        elif isinstance(module, QuantizedLinear4bit):
            quantized_count += 1

    return linear_found, quantized_count
