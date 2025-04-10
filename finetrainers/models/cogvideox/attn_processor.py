import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
import matplotlib.pyplot as plt
import numpy as np

class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        apply_target_noise_only: str = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        if apply_target_noise_only:
            # query.shape : (B, num_heads, Tk, head_dim)
            num_frames = 13
            height = 32
            width = 48
            
            # Calculate number of frame blocks
            full_length = query.shape[2]
            image_length = full_length - text_seq_length
            num_blocks = image_length // num_frames
            
            # Split query, key, value into heads
            query_heads = query.chunk(attn.heads, dim=1)  # List of (B, 1, Tk, head_dim)
            key_heads = key.chunk(attn.heads, dim=1)
            value_heads = value.chunk(attn.heads, dim=1)
            
            # Compute attention for each head
            hidden_states_heads = []
            # print("Processing attention for each head...")
            for head_idx, (q_head, k_head, v_head) in enumerate(zip(query_heads, key_heads, value_heads)):
                # Create mask for this head
                with torch.no_grad():
                    head_mask = torch.ones(full_length, full_length, device=q_head.device, requires_grad=False)
                    
                    if apply_target_noise_only == "front":
                        target_start = text_seq_length
                        target_end = text_seq_length + num_frames
                        head_mask[target_start:target_end, :target_start] = 0
                        head_mask[target_start:target_end, target_end:] = 0
                        
                    elif apply_target_noise_only == "back":
                        target_start = text_seq_length + (num_blocks - 1) * num_frames
                        target_end = text_seq_length + num_blocks * num_frames
                        head_mask[target_start:target_end, :target_start] = 0
                        head_mask[target_start:target_end, :target_start] = 0
                    
                    # Add batch dimension
                    head_mask = head_mask.unsqueeze(0)
                
                # Compute attention for this head
                head_hidden_states = F.scaled_dot_product_attention(
                    q_head, k_head, v_head,
                    attn_mask=head_mask,
                    dropout_p=0.0,
                    is_causal=False
                )
                
                # Store result and clear memory
                hidden_states_heads.append(head_hidden_states)
                del head_mask, q_head, k_head, v_head
            
            # Concatenate all heads
            hidden_states = torch.cat(hidden_states_heads, dim=1)
            
            # Clear GPU memory
            del hidden_states_heads, query_heads, key_heads, value_heads
            torch.cuda.empty_cache()
        else:
            # Original attention computation without masking
            hidden_states = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states