# SonicMoE patches

Two diffs against [SonicMoE](https://github.com/Dao-AILab/sonic-moe), applied
by `setup.sh` after it clones the repo.

**They are pinned to SonicMoE commit `66f6ced475259d9aa9ebe854652784b2a45409bd`**
(2026-02-12). `setup.sh` checks out that exact revision. Do not remove the pin:
SonicMoE has since restructured both files these patches touch, and against
current `main` neither applies.

| Patch | What it changes | Needed for |
|---|---|---|
| `sonicmoe_mixed_precision.patch` | Casts `router_w`, `w1`, `w2`, `b1`, `b2` to the input dtype in `sonicmoe/functional/__init__.py` before the router `F.linear` | bf16 / autocast training. Without it the kernel fails on a dtype mismatch. |
| `sonicmoe_grouped_loss.patch` | Stores `_routing_acc_probs` and `_routing_expert_freq` on the `MoE` module in `sonicmoe/moe.py` | Grouped auxiliary loss. `more/mixtral/modeling_mixtral.py` reads both attributes when `use_grouped_aux_loss=true`, which the shipped Mixtral config sets. |

## If a patch stops applying

`setup.sh` prints a `*** PATCH FAILED ***` block and continues, because the
DeepSeek-v3 path does not use SonicMoE. But Mixtral + SonicMoE training will
crash — for the grouped-loss patch, with:

```
AttributeError: 'MoE' object has no attribute '_routing_acc_probs'
```

To re-target a patch at a newer SonicMoE, apply the change by hand in a clean
checkout and regenerate with `git diff > patches/<name>.patch`, then update
`SONICMOE_COMMIT` in `setup.sh`. Regenerate rather than hand-editing hunk
headers: both of these patches previously shipped with line counts that were
off by one, so `git apply` rejected them as `corrupt patch` at any commit.

## Verifying

```bash
cd sonic-moe
git apply --check ../patches/sonicmoe_grouped_loss.patch      # applies
git apply --reverse --check ../patches/sonicmoe_grouped_loss.patch  # already applied
```

A failure from both means the checkout does not match the patch.
