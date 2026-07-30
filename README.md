# MoRE: Mixture of Reused Experts

Reference implementation of Paper _MoRE: Mixture of Reused Experts_ (COLM 2026).

## What MoRE is

Three changes to a standard Mixture-of-Experts layer, which appear directly as
config fields:

| Field                 | Meaning                                                    |
| --------------------- | ---------------------------------------------------------- |
| `grouping_factor`     | the reuse factor R — how many layers share one expert pool |
| `tying_strategy`      | `"block"` (adjacent layers) or `"strided"`                 |
| `use_depth_embedding` | learnable per-layer embedding added before routing         |

Implemented on three backbones: DeepSeek-v3, Qwen3-MoE, and Mixtral.

## Install

### Inspect the models — no GPU

```bash
git clone https://github.com/EricQiu6/MoRE.git && cd MoRE
pip install -r requirements.txt

# Patch whichever backbones you want to use.
python apply_patch.py --model deepseek_v3
python apply_patch.py --model qwen3_moe
python apply_patch.py --model mixtral
```

Then run the **Quickstart** below to confirm the install works.

`apply_patch.py` copies MoRE's model files over your installed `transformers`
package. It backs up the originals first:

```bash
python apply_patch.py --check      # which families are patched
python apply_patch.py --restore    # put the originals back
```

All three families live in separate directories inside `transformers`, so any
subset can be patched at once and they never interfere.

Requires `transformers==5.1.0` exactly — the modeling code targets that
version's RoPE and key-mapping APIs, and `apply_patch.py` refuses to run
against anything else.

### Train — Hopper GPU (H100/H200), CUDA 12.9+, Python 3.12+

```bash
bash setup.sh
```

Creates a venv, installs the CUDA 12.9 build of PyTorch, builds SonicMoE from
source with our two patches, patches transformers, and prepares the tokenizer.
It exits early if no NVIDIA GPU is present, since the CUDA wheel index it uses
publishes no CPU or macOS builds.

The paper's runs used this container image, which is the reference environment
for the training path:

```
runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204
```

## Quickstart

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("configs/example/more-deepseek_v3-114m")
model = AutoModelForCausalLM.from_config(cfg)

ids = torch.randint(0, cfg.vocab_size, (1, 128))
print(model(input_ids=ids, labels=ids).loss)

# Tied experts: adjacent layers hold the SAME tensor, so the total parameter
# count exceeds the number of unique parameters. remove_duplicate=False is
# required -- the default walk de-duplicates by identity and would report the
# two numbers as equal for every model, tied or not.
seen, total, unique = set(), 0, 0
for _, p in model.named_parameters(remove_duplicate=False):
    total += p.numel()
    if p.data_ptr() not in seen:
        seen.add(p.data_ptr())
        unique += p.numel()
print(f"{total:,} total, {unique:,} unique")
# 194,551,296 total, 114,925,056 unique
```

## Example configs

| Directory                               | Backbone    |
| --------------------------------------- | ----------- |
| `configs/example/more-deepseek_v3-114m` | DeepSeek-v3 |
| `configs/example/more-qwen3_moe-118m`   | Qwen3-MoE   |
| `configs/example/more-mixtral-118m`     | Mixtral     |

The generators in `scripts/` are the source of truth; the committed JSON is
their output. Running a generator with no arguments reproduces the committed
config exactly.

```bash
python scripts/gen_conf_deepseek_v3.py --preset more-114m --out /tmp/my-config
```

Each generator also carries a `baseline-*` preset — the matched MoE baseline
with `grouping_factor=1` and no depth embedding.

The Mixtral example ships with `use_scattermoe: false` so it builds anywhere.
Pass `--sonic` to emit the SonicMoE training variant, which needs a Hopper GPU.

## Data

The dataset was built in two steps:

1. The MoEUT codebase ([https://github.com/RobertCsordas/moeut](https://github.com/RobertCsordas/moeut)) tokenizes C4
   with an 8000-token SentencePiece vocabulary into `.bin` chunks of `int16`
   token IDs.
2. `scripts/gen_tok_data_c4.py` reshapes those chunks into 1024-token windows
   and saves a HuggingFace dataset.

To train on your own data, produce a dataset in HuggingFace `save_to_disk`
format with these columns:

| Column           | Type              | Value                         |
| ---------------- | ----------------- | ----------------------------- |
| `input_ids`      | `Sequence(int32)` | `block_size` token IDs (1024) |
| `attention_mask` | `Sequence(int8)`  | all ones                      |
| `labels`         | `Sequence(int32)` | equal to`input_ids`           |

`block_size` must equal the model's `max_position_embeddings`, which is 1024
in all three example configs. The tokenizer used for the paper is in
`c4_tokenizer/`.

## Training

```bash
python scripts/train_c4_sonic_single.py \
    --model_name_or_path configs/example/more-deepseek_v3-114m \
    --train_data /path/to/train \
    --val_data /path/to/val \
    --batch_size 64 --max_steps 100000
```

Multi-GPU uses the same script under `torchrun`

```bash
torchrun --nproc_per_node=4 scripts/train_c4_sonic_single.py \
    --model_name_or_path configs/example/more-deepseek_v3-114m \
    --train_data /path/to/train \
    --val_data /path/to/val \
    --batch_size 16 --max_steps 100000 --eval_steps 10000
```

### MoEUT baseline

`more/moeut_hf/` wraps MoEUT rather than bundling it:

```bash
git clone https://github.com/RobertCsordas/moeut
export MOEUT_PATH=/path/to/moeut

torchrun --nproc_per_node=4 scripts/train_c4_sonic_single.py \
    --model_type moeut --moeut_profile MoEUT_126M \
    --train_data /path/to/train --val_data /path/to/val \
    --batch_size 16 --max_steps 100000
```

`--moeut_profile` must name a profile defined in that repo's `profiles.py`.

## Repository map

```
more/            model implementations (deepseek_v3, qwen3_moe, mixtral, moeut_hf)
configs/example/ three example configs
scripts/         config generators, training, data conversion
patches/         SonicMoE patches applied by setup.sh
```

`more/moeut_hf/` wraps the MoEUT baseline, which is **not** bundled. Clone it
and set `MOEUT_PATH` to use it.

## Authors

Eric S. Qiu, Utku Umur Acikalin, Justin Lovelace, Christian Belardi,
Arjun B. Mulchandani, Carla P. Gomes, Kilian Q. Weinberger

## Citation

```bibtex
@inproceedings{qiu2026more,
  title     = {MoRE: Mixture of Reused Experts},
  author    = {Qiu, Eric S. and Acikalin, Utku Umur and Lovelace, Justin and
               Belardi, Christian and Mulchandani, Arjun B. and Gomes, Carla P. and
               Weinberger, Kilian Q.},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Apache-2.0 — see `LICENSE`. Third-party components and their licenses are
listed in `NOTICE`.
