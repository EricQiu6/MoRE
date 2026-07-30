"""Generate the MoRE Qwen3-MoE example config.

Qwen3-MoE has no shared experts, so only routed experts are tied. Otherwise
the MoRE knobs are identical to the DeepSeek-v3 generator; see that file for
the design rationale behind keeping presets in code rather than as loose JSON.

Requires `python apply_patch.py --model qwen3_moe` first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

ROPE = {"rope_type": "default", "rope_theta": 10000.0}

# The alternative-backbone experiment from the paper: MoRE on a Qwen3-MoE
# backbone at the same 768-wide, 18-layer shape as the DeepSeek runs.
#
# Expert matching mirrors DeepSeek's: the baseline gets 12 experts per layer
# (18 x 12 = 216 expert slots), and MoRE (R=2) gets 24 shared across each
# adjacent pair (9 pools x 24 = 216), so both hold the same number of unique
# physical experts and both measure ~118M unique parameters.
PRESETS: dict[str, dict] = {
    "more-118m": dict(
        vocab_size=8000,
        hidden_size=768,
        intermediate_size=2048,          # dense-layer width
        moe_intermediate_size=128,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_parameters=ROPE,
        decoder_sparse_step=1,
        num_experts_per_tok=4,
        num_experts=24,                  # 2 x the baseline's 12
        norm_topk_prob=True,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
        use_depth_embedding=True,
        grouping_factor=2,
        tying_strategy="block",
    ),
    # The MoE baseline: 12 experts, no sharing, no depth embedding. Build both
    # and compare the unique counts from count_parameters below -- they should
    # match, since R=2 over 2n experts holds the same unique pool as n per layer.
    "baseline-118m": dict(
        vocab_size=8000,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=128,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_parameters=ROPE,
        decoder_sparse_step=1,
        num_experts_per_tok=4,
        num_experts=12,
        norm_topk_prob=True,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
        use_depth_embedding=False,
        grouping_factor=1,
        tying_strategy="block",
    ),
}


def count_parameters(model) -> tuple[int, int]:
    """Return (total, unique).

    remove_duplicate=False is load-bearing: expert tying assigns the SAME
    nn.Parameter object to several layers, and the default walk de-duplicates
    by identity, reporting total == unique for every model. See the
    DeepSeek-v3 generator for the full rationale.
    """
    seen: set[int] = set()
    total = unique = 0
    for _name, p in model.named_parameters(remove_duplicate=False):
        total += p.numel()
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique += p.numel()
    return total, unique


def build_config(preset: str) -> Qwen3MoeConfig:
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    return Qwen3MoeConfig(**PRESETS[preset])


def build_model(preset: str) -> Qwen3MoeForCausalLM:
    return Qwen3MoeForCausalLM(build_config(preset))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="more-118m", choices=sorted(PRESETS))
    ap.add_argument("--out", default="configs/example/more-qwen3_moe-118m")
    args = ap.parse_args()

    model = build_model(args.preset)
    total, unique = count_parameters(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(out)
    model.generation_config.save_pretrained(out)
    print(f"{args.preset}: {total:,} total params, {unique:,} unique -> {out}")


if __name__ == "__main__":
    main()
