"""
NVFP4 Block-Scaled Group GEMM for B200 - v2
"""

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled group GEMM kernel.
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []

    # Pre-extract all tensors to minimize Python overhead in hot loop
    num_groups = len(abc_tensors)

    for i in range(num_groups):
        a, b, c = abc_tensors[i]
        sfa_reordered, sfb_reordered = sfasfb_reordered_tensors[i]

        # L=1 always, so access directly without loop
        scale_a = sfa_reordered.reshape(-1)
        scale_b = sfb_reordered.reshape(-1)

        # Get matrix slices (L=1, so index 0)
        a_mat = a[:, :, 0].view(torch.float4_e2m1fn_x2)
        b_mat = b[:, :, 0].T.view(torch.float4_e2m1fn_x2)

        # Execute scaled matrix multiplication
        c[:, :, 0] = torch._scaled_mm(
            a_mat,
            b_mat,
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )

        result_tensors.append(c)

    return result_tensors
