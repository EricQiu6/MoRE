"""
HuggingFace-style wrapper for MoEUT (clean repo: github.com/RobertCsordas/moeut).
Wraps MoEUTLM so it matches our Trainer: forward(input_ids, attention_mask, labels) -> loss, logits, aux_loss.
"""
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class MoEUTLMForCausalLM(torch.nn.Module):
    """Wraps MoEUTLM so it matches our Trainer (HF-style inputs, .loss, .aux_loss, .router_aux_loss_coef)."""

    def __init__(self, vocab_size: int, router_aux_loss_coef: float = 1.0, **moeut_kw):
        super().__init__()
        from moeut import MoEUTLM
        self.moeut = MoEUTLM(n_tokens=vocab_size, **moeut_kw)
        self.router_aux_loss_coef = router_aux_loss_coef

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Causal: logits at t predict token at t+1. MoEUTLM expects [B, T], returns [B, T, V].
        out = self.moeut(input_ids)
        logits = out.outputs  # [B, T, V]
        V = logits.size(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(-1, V)
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous().view(-1)
        else:
            shift_labels = input_ids[..., 1:].contiguous().view(-1)
        nll = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        total_loss = nll + self.router_aux_loss_coef * out.reg_loss
        output = CausalLMOutputWithPast(loss=total_loss, logits=logits)
        setattr(output, "aux_loss", out.reg_loss)
        return output
