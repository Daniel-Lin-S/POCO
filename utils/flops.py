"""Count forward and backward FLOPs of one actual training step.

Input is a differentiable step callable. Output is a JSON-ready record with
``forward_backward_flops_per_step`` and profiling details. The count uses
PyTorch's operator FLOP formulas plus the attention matrix products in
xFormers block-diagonal attention. It excludes optimizer work and scalar
non-matrix operations not assigned FLOP formulas by PyTorch.
"""

from collections.abc import Callable

from torch.utils.flop_counter import FlopCounterMode

ATTENTION_FORWARD_BACKWARD_MULTIPLIER = 12


def measure_train_step(step: Callable[[], object]) -> tuple[dict, object]:
    """Profile one forward and backward pass without changing the optimizer.

    Parameters
    ----------
    step : Callable[[], object]
        Runs weighted training loss and ``backward()`` on an actual batch.

    Returns
    -------
    tuple[dict, object]
        FLOP record and original step result.
    """
    import models.poyo.rotary_attention as rotary

    attention_flops = 0
    fused_counted = False
    counter = FlopCounterMode(display=False)
    original = (
        rotary.xops.memory_efficient_attention
        if rotary.xops else None
    )

    def counted_attention(q, k, v, attn_bias=None, p=0.0):
        nonlocal attention_flops, fused_counted
        if attn_bias is None or not hasattr(attn_bias, 'q_seqinfo'):
            raise ValueError(
                'FLOP counter requires block-diagonal xFormers attention.'
            )
        q_starts = attn_bias.q_seqinfo.seqstart_py
        k_starts = attn_bias.k_seqinfo.seqstart_py
        q_lengths = [b - a for a, b in zip(q_starts, q_starts[1:])]
        k_lengths = [b - a for a, b in zip(k_starts, k_starts[1:])]
        products = sum(q_len * k_len for q_len, k_len in
                       zip(q_lengths, k_lengths))
        before = counter.get_total_flops()
        output = original(q, k, v, attn_bias=attn_bias, p=p)
        if counter.get_total_flops() > before:
            fused_counted = True
        else:
            attention_flops += (
                ATTENTION_FORWARD_BACKWARD_MULTIPLIER
                * products * q.shape[2] * q.shape[3]
            )
        return output

    if original is not None:
        rotary.xops.memory_efficient_attention = counted_attention
    try:
        with counter:
            step_result = step()
        operator_flops = counter.get_total_flops()
    finally:
        if original is not None:
            rotary.xops.memory_efficient_attention = original
    result = int(operator_flops + attention_flops)
    if result <= 0:
        raise ValueError('No forward or backward FLOPs were counted.')
    return {
        'forward_backward_flops_per_step': result,
        'pytorch_operator_flops': int(operator_flops),
        'xformers_attention_flops': int(attention_flops),
        'fused_attention_counted_by_pytorch': fused_counted,
        'method': 'torch_flop_counter_with_unmapped_attention_estimate',
        'scope': 'first actual training step, excluding optimizer',
        'measured_step': 1,
        'counted_ops_only': True,
    }, step_result
