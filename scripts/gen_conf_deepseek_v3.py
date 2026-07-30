"""Generate the MoRE DeepSeek-v3 example config.

Design decisions:
  * Presets live in this file, not as loose JSON. The generator is the single
    source of truth; configs/example/ holds its output.
  * The three MoRE knobs are use_depth_embedding (learned per-layer embedding
    added before routing), grouping_factor (the reuse factor R — how many
    adjacent layers share one expert pool), and tying_strategy ("block" or
    "strided").
  * Requires `python apply_patch.py --model deepseek_v3` first, so that
    DeepseekV3Config carries the MoRE fields.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

ROPE = {"rope_type": "default", "rope_theta": 10000.0}

# Transcribed from the paper's Table "Architectural specifications", Small
# row. Reading the table into DeepseekV3Config fields:
#   d_model 768, L 18, H 12, d_FFN 128        -> hidden/layers/heads/moe_inter
#   d_head "64, 64" = (qkhead, vhead)         -> qk head total 64, v_head_dim 64
#   d_rope, d_nope "32, 32"                   -> the split WITHIN the 64-wide
#                                                qk head: 32 + 32 = 64
#   d_c^q, d_c^kv "256, 128"                  -> q_lora_rank, kv_lora_rank
#   N "29+1", K "3+1"                         -> 29 routed + 1 always-on shared,
#                                                3 routed active + the shared one
#
# Why 29 routed: the paper matches MoRE to DeepSeek by unique physical
# experts. DeepSeek has 14+1 = 15 per layer over 18 layers; MoRE (R=2) shares
# one pool of 30 across each adjacent pair, so 2 x 15 = 30 = 29+1. Both land
# at ~114M unique parameters, which is what the table's "Param." column
# reports — unique, not total.
#
# Several DeepseekV3Config defaults are actively wrong here and must be set:
#   n_group defaults to 8, but 29 is not divisible by 8, so grouped top-k
#     routing reshapes to an invalid shape and throws.
#   first_k_dense_replace defaults to 3, which would silently turn the first
#     three MoE layers into dense MLPs.
#   q_lora_rank defaults to 1536 against the paper's 256.
PRESETS: dict[str, dict] = {
    "more-114m": dict(
        vocab_size=8000,
        hidden_size=768,
        intermediate_size=2048,          # dense-layer width (unused: no dense layers)
        moe_intermediate_size=128,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=29,             # +1 shared = 2 x DeepSeek's 15
        routed_scaling_factor=1.0,
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_group=1,                       # 29 is not divisible by 8; must be 1
        topk_group=1,                    # follows from n_group=1
        num_experts_per_tok=3,           # +1 shared = 4 active
        first_k_dense_replace=0,         # no dense layers
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_parameters=ROPE,
        use_depth_embedding=True,
        grouping_factor=2,
        tying_strategy="block",
    ),
    # The DeepSeek baseline from the same table row: 14+1 experts, no sharing,
    # no depth embedding. Matched to MoRE at ~114M unique parameters, which is
    # the paper's parameter-matching claim: build both and compare the unique
    # counts from count_parameters below.
    "baseline-114m": dict(
        vocab_size=8000,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=128,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=14,
        routed_scaling_factor=1.0,
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=3,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_parameters=ROPE,
        use_depth_embedding=False,
        grouping_factor=1,
        tying_strategy="block",
    ),
}


def count_parameters(model) -> tuple[int, int]:
    """Return (total, unique).

    Design decision: remove_duplicate=False is load-bearing. Expert tying
    assigns the SAME nn.Parameter object to several layers, and
    named_parameters() de-duplicates by object identity by default — so the
    default walk yields each shared tensor once and reports total == unique
    for every model, tied or not. The working repo's copy of this function
    has that bug; do not reintroduce it.
    """
    seen: set[int] = set()
    total = unique = 0
    for _name, p in model.named_parameters(remove_duplicate=False):
        total += p.numel()
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique += p.numel()
    return total, unique


def build_config(preset: str) -> DeepseekV3Config:
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    return DeepseekV3Config(**PRESETS[preset])


def build_model(preset: str) -> DeepseekV3ForCausalLM:
    return DeepseekV3ForCausalLM(build_config(preset))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="more-114m", choices=sorted(PRESETS))
    ap.add_argument(
        "--out", default="configs/example/more-deepseek_v3-114m",
        help="directory to write config.json and generation_config.json into",
    )
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
