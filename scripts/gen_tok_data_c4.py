"""Convert MoEUT-tokenized .bin chunks into a HuggingFace dataset.

This is the record of how the paper's dataset was built. The processed
dataset itself is not published (see README, "Data"), and the upstream
tokenization step lives in the MoEUT codebase, not here. You need .bin
chunks of int16 token IDs to use this.

Output format -- this is what train_c4_sonic_single.py expects, and what you
must match if you bring your own data:
    input_ids      Sequence(int32), length block_size
    attention_mask Sequence(int8),  all ones
    labels         Sequence(int32), equal to input_ids (causal LM)
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from datasets import Dataset, Features, Sequence, Value

# MoEUT writes int16; reading as another dtype silently garbles every token.
DATA_DTYPE = np.int16

# Context length. Must equal the model's max_position_embeddings -- all three
# example configs use 1024, which is the paper's context length. A larger
# value produces examples the model has no position embeddings for.
DEFAULT_BLOCK_SIZE = 1024


def binary_dataset_generator(input_dir: str, pattern: str, block_size: int):
    """Read raw binary files and yield fixed-size context windows.

    Tokens are a flat stream, so each file is chopped into floor(n/block_size)
    windows and the remainder is dropped.
    """
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching {pattern}")
    print(f"Found {len(files)} binary files. Processing...")

    for filepath in files:
        print(f"Processing {filepath}...")
        try:
            # memmap so we never load a whole chunk into RAM.
            raw_data = np.memmap(filepath, dtype=DATA_DTYPE, mode="r")
        except (OSError, ValueError) as e:
            print(f"Error reading {filepath}: {e}")
            continue

        n_samples = len(raw_data) // block_size
        if n_samples == 0:
            continue
        reshaped = raw_data[: n_samples * block_size].reshape(n_samples, block_size)
        for i in range(n_samples):
            ids = reshaped[i].tolist()
            yield {
                "input_ids": ids,
                "attention_mask": [1] * block_size,
                "labels": ids,
            }
        del raw_data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input-dir", required=True,
        help="directory of MoEUT .bin tokenized chunks",
    )
    ap.add_argument(
        "--output", required=True,
        help="destination directory for datasets.save_to_disk",
    )
    ap.add_argument("--pattern", default="*.bin")
    ap.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help=(
            "tokens per training example; must equal the model's "
            f"max_position_embeddings (default {DEFAULT_BLOCK_SIZE})"
        ),
    )
    args = ap.parse_args(argv)

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")),
    })
    ds = Dataset.from_generator(
        binary_dataset_generator,
        features=features,
        gen_kwargs={
            "input_dir": args.input_dir,
            "pattern": args.pattern,
            "block_size": args.block_size,
        },
    )
    print(f"Dataset created. Rows: {len(ds)}")
    ds.save_to_disk(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
