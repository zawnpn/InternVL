import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from transformers.models.qwen3.modeling_utils import ALL_ATTENTION_FUNCTIONS

def qwen3_flash_attention_2_for_packed_training(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout,
    scaling,
    sliding_window,
    *args,
    **kwargs,
):
    """
    Packed-training variant of the `flash_attention_2` implementation for Qwen3.

    module: the Qwen3Attention instance
    query_states: Tensor of shape (1, num_heads, total_seq_len, head_dim)
    key_states:   Tensor of same shape
    value_states: Tensor of same shape
    attention_mask:  Either a cumulative-lengths Tensor of shape (1, num_spans+1) or a boolean mask (1, seq_len)
    dropout:       attention dropout probability
    scaling:       scaling factor for QK^T (i.e., 1/sqrt(head_dim))
    sliding_window: size of sliding window or None for global
    """
    # Expect batch size = 1 for packed training
    assert query_states.size(0) == 1, "Packed training requires batch size of 1"
    # remove batch dim for q, k, v
    q = query_states.squeeze(0)
    k = key_states.squeeze(0)
    v = value_states.squeeze(0)

    # build cumulative sequence lengths
    if attention_mask.dim() == 2 and attention_mask.dtype == torch.bool:
        # single-span case: count True tokens
        mask = attention_mask.squeeze(0)
        seqlen = int(mask.long().sum().item())
        cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=mask.device)
    else:
        # assume already cu_seqlens in int form
        cu_seqlens = attention_mask.squeeze(0).to(torch.int32)

    # compute max span length
    span_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(span_lengths.max().item())

    # determine causal flag
    causal = getattr(module, "is_causal", True)

    # choose window_size: fixed window or global
    if sliding_window is None:
        window_size = (-1, -1)
    else:
        window_size = (sliding_window, sliding_window)

    # invoke flash_attn varlen API
    attn_out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=causal,
        window_size=window_size,
    )

    # restore batch dim and return dummy weights
    return attn_out.unsqueeze(0), None

# override the default flash_attention_2 implementation for Qwen3
ALL_ATTENTION_FUNCTIONS.register(
    "flash_attention_2", qwen3_flash_attention_2_for_packed_training
)
print("Registered `flash_attention_2` for packed training in Qwen3.")
