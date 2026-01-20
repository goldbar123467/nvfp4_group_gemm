import torch
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    result_tensors = []
    for (a, b, c), (sfa_reordered, sfb_reordered) in zip(abc_tensors, sfasfb_reordered_tensors):
        scale_a = sfa_reordered.reshape(-1)
        scale_b = sfb_reordered.reshape(-1)
        result = torch._scaled_mm(a[:, :, 0].view(torch.float4_e2m1fn_x2), b[:, :, 0].T.view(torch.float4_e2m1fn_x2), scale_a, scale_b, bias=None, out_dtype=torch.float16)
        c[:, :, 0] = result
        result_tensors.append(c)
    return result_tensors
