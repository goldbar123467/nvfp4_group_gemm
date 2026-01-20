"""
NVFP4 Block-Scaled Group GEMM for B200 - v6 with CUDA graphs
"""

import torch
from task import input_t, output_t

# Single GEMM operation - will be compiled
@torch.compile(mode="reduce-overhead")
def _single_gemm(a_mat, b_mat, scale_a, scale_b):
    return torch._scaled_mm(
        a_mat,
        b_mat,
        scale_a,
        scale_b,
        bias=None,
        out_dtype=torch.float16,
    )


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled group GEMM kernel with torch.compile.
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []

    for (a, b, c), (sfa_reordered, sfb_reordered) in zip(
        abc_tensors, sfasfb_reordered_tensors
    ):
        scale_a = sfa_reordered.reshape(-1)
        scale_b = sfb_reordered.reshape(-1)

        a_mat = a[:, :, 0].view(torch.float4_e2m1fn_x2)
        b_mat = b[:, :, 0].T.view(torch.float4_e2m1fn_x2)

        result = _single_gemm(a_mat, b_mat, scale_a, scale_b)
        c[:, :, 0] = result
        result_tensors.append(c)

    return result_tensors
