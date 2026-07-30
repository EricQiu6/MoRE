"""Generate the MoRE Mixtral example config.

Design decision: use_scattermoe defaults to False in the shipped example so
the config builds on any machine. SonicMoE needs a Hopper GPU (H100/H200,
CUDA 12.9+); the reference MixtralSparseMoeBlock is the same computation and
runs on CPU. Pass --sonic to emit the training variant.

Requires `python apply_patch.py --model mixtral` first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import MixtralConfig, MixtralForCausalLM

# The alternative-backbone experiment from the paper: MoRE on a Mixtral
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
        intermediate_size=128,           # expert width (Mixtral has no dense MLP)
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=12,
        num_local_experts=24,            # 2 x the baseline's 12
        num_experts_per_tok=4,
        rms_norm_eps=1e-5,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        use_grouped_aux_loss=True,
        use_depth_embedding=True,
        grouping_factor=2,
        tying_strategy="block",
        use_scattermoe=False,
    ),
    # The MoE baseline: 12 experts, no sharing, no depth embedding. Build both
    # and compare the unique counts from count_parameters below -- they should
    # match, since R=2 over 2n experts holds the same unique pool as n per layer.
    "baseline-118m": dict(
        vocab_size=8000,
        hidden_size=768,
        intermediate_size=128,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=12,
        num_local_experts=12,
        num_experts_per_tok=4,
        rms_norm_eps=1e-5,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        use_grouped_aux_loss=True,
        use_depth_embedding=False,
        grouping_factor=1,
        tying_strategy="block",
        use_scattermoe=False,
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


def build_config(preset: str, sonic: bool = False) -> MixtralConfig:
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    kwargs = dict(PRESETS[preset])
    kwargs["use_scattermoe"] = sonic
    return MixtralConfig(**kwargs)


def build_model(preset: str, sonic: bool = False) -> MixtralForCausalLM:
    return MixtralForCausalLM(build_config(preset, sonic))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="more-118m", choices=sorted(PRESETS))
    ap.add_argument("--out", default="configs/example/more-mixtral-118m")
    ap.add_argument(
        "--sonic", action="store_true",
        help="emit the SonicMoE training variant (requires a Hopper GPU)",
    )
    args = ap.parse_args()

    model = build_model(args.preset, args.sonic)
    total, unique = count_parameters(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(out)
    model.generation_config.save_pretrained(out)
    print(f"{args.preset}: {total:,} total params, {unique:,} unique -> {out}")


if __name__ == "__main__":
    main()
