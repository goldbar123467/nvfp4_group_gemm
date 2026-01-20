"""
NVFP4 Block-Scaled Group GEMM for NVIDIA B200
"""

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled group GEMM kernel.
    Uses pre-reordered scale factors to avoid recomputation.
    """
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []

    for (a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l) in zip(
        abc_tensors, sfasfb_reordered_tensors, problem_sizes
    ):
        for l_idx in range(l):
            # Use pre-reordered scale factors (already in blocked format on GPU)
            scale_a = sfa_reordered[:, :, :, :, :, l_idx].contiguous().flatten()
            scale_b = sfb_reordered[:, :, :, :, :, l_idx].contiguous().flatten()

            # Execute scaled matrix multiplication
            result = torch._scaled_mm(
                a[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a,
                scale_b,
                bias=None,
                out_dtype=torch.float16,
            )
            c[:, :, l_idx] = result

        result_tensors.append(c)

    return result_tensors
