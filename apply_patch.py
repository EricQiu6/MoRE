"""Install MoRE's patched model files into an existing transformers package.

Why this exists: MoRE modifies three HuggingFace model implementations
(depth embeddings + expert tying). Rather than vendoring a whole fork, we
copy the three trees over the installed package.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REQUIRED_TRANSFORMERS = "5.1.0"

REPO_ROOT = Path(__file__).resolve().parent

MODEL_FILES: dict[str, list[str]] = {
    "deepseek_v3": [
        "configuration_deepseek_v3.py",
        "modeling_deepseek_v3.py",
    ],
    "qwen3_moe": [
        "configuration_qwen3_moe.py",
        "modeling_qwen3_moe.py",
    ],
    "mixtral": [
        "configuration_mixtral.py",
        "modeling_mixtral.py",
        "modular_mixtral.py",
    ],
}


class PatchError(RuntimeError):
    """Base class for patch failures."""


class AlreadyPatched(PatchError):
    pass


class VersionMismatch(PatchError):
    pass


# Tests point this at a tmp directory so they never read or delete the real
# backup. Without the override, running the suite on a patched install would
# wipe the manifest and leave --restore unable to recover the originals.
_BACKUP_DIR_OVERRIDE: Path | None = None


def backup_dir() -> Path:
    if _BACKUP_DIR_OVERRIDE is not None:
        return Path(_BACKUP_DIR_OVERRIDE)
    return REPO_ROOT / ".patch_backup"


def _manifest_path() -> Path:
    return backup_dir() / "manifest.json"


def read_manifest() -> dict:
    if not _manifest_path().exists():
        return {}
    return json.loads(_manifest_path().read_text())


def _write_manifest(data: dict) -> None:
    backup_dir().mkdir(exist_ok=True)
    _manifest_path().write_text(json.dumps(data, indent=2))


def installed_version() -> str:
    import transformers

    return transformers.__version__


def transformers_root() -> Path:
    import transformers

    return Path(transformers.__file__).parent


def apply(model: str, root: Path | None = None, check_version: bool = False) -> None:
    """Copy this repo's `model` tree over the installed transformers copy."""
    if model not in MODEL_FILES:
        raise PatchError(f"unknown model {model!r}; choose from {list(MODEL_FILES)}")

    if check_version:
        found = installed_version()
        if found != REQUIRED_TRANSFORMERS:
            raise VersionMismatch(
                f"MoRE requires transformers=={REQUIRED_TRANSFORMERS}, found {found}. "
                f"The modeling code targets {REQUIRED_TRANSFORMERS}'s RoPE and "
                f"key-mapping APIs. Install the pin: "
                f"pip install transformers=={REQUIRED_TRANSFORMERS}"
            )

    root = Path(root) if root is not None else transformers_root()
    dest = root / "models" / model
    if not dest.is_dir():
        raise PatchError(
            f"{dest} not found. Your transformers install does not include "
            f"the {model} model."
        )

    manifest = read_manifest()
    if model in manifest:
        raise AlreadyPatched(
            f"{model} is already patched. Run --restore first if you want to "
            f"re-apply."
        )

    saved = backup_dir() / model
    saved.mkdir(parents=True, exist_ok=True)
    for fname in MODEL_FILES[model]:
        src = REPO_ROOT / "more" / model / fname
        if not src.exists():
            raise PatchError(f"missing source file {src}")
        original = dest / fname
        if original.exists():
            shutil.copy2(original, saved / fname)
        shutil.copy2(src, original)

    manifest[model] = {"dest": str(dest), "files": MODEL_FILES[model]}
    _write_manifest(manifest)


def restore(root: Path | None = None) -> None:
    """Put every backed-up original back. Safe to call repeatedly."""
    manifest = read_manifest()
    if not manifest:
        return
    for model, info in list(manifest.items()):
        dest = Path(info["dest"]) if root is None else Path(root) / "models" / model
        for fname in info["files"]:
            saved = backup_dir() / model / fname
            if saved.exists():
                shutil.copy2(saved, dest / fname)
        del manifest[model]
    _write_manifest(manifest)
    shutil.rmtree(backup_dir(), ignore_errors=True)


def status(root: Path | None = None) -> dict[str, bool]:
    manifest = read_manifest()
    return {model: model in manifest for model in MODEL_FILES}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODEL_FILES))
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        for model, patched in status().items():
            print(f"{model:14s} {'PATCHED' if patched else 'original'}")
        return 0
    if args.restore:
        restore()
        print("Restored original transformers files.")
        return 0
    if not args.model:
        ap.error("one of --model, --restore, or --check is required")

    try:
        apply(args.model, check_version=True)
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Patched {args.model}. Undo with: python apply_patch.py --restore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
