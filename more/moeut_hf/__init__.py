"""Adapters for the MoEUT baseline.

MoEUT itself is NOT bundled here. This package wraps Robert Csordas's
implementation (https://github.com/RobertCsordas/moeut, MIT) so it plugs
into our Trainer.
"""
from .moeut_for_causal_lm import MoEUTLMForCausalLM

__all__ = [
    "MoEUTLMForCausalLM",
    "MoEUTNotFound",
    "get_moeut_profile",
]


class MoEUTNotFound(ImportError):
    """Raised when the external MoEUT repository cannot be located.

    Design decision: MoEUT is a runtime dependency the reader must clone, not
    something we vendor. MOEUT_PATH previously had a hardcoded default
    pointing at the authors' own machine, which failed elsewhere with an
    opaque ModuleNotFoundError about `profiles`. There is no default now:
    unset means unset, and the error says what to do.
    """


def get_moeut_profile(name: str):
    """Resolve a MoEUT profile name to kwargs. Requires the MoEUT repo."""
    import os
    import sys

    moeut_root = os.environ.get("MOEUT_PATH")
    if not moeut_root or not os.path.isdir(moeut_root):
        raise MoEUTNotFound(
            "MoEUT baseline code not found. MoEUT is a separate project and is "
            "not bundled here.\n"
            "  1. git clone https://github.com/RobertCsordas/moeut\n"
            "  2. export MOEUT_PATH=/path/to/moeut\n"
            f"(MOEUT_PATH is currently {moeut_root!r})"
        )
    if moeut_root not in sys.path:
        sys.path.insert(0, moeut_root)
    import profiles

    if not hasattr(profiles, name):
        available = sorted(
            n for n in dir(profiles)
            if n.startswith("MoEUT") and isinstance(getattr(profiles, n), dict)
        )
        raise MoEUTNotFound(
            f"MoEUT profile {name!r} is not defined in {moeut_root}/profiles.py.\n"
            f"Available upstream: {', '.join(available) or '(none found)'}\n"
            "\n"
            "The paper's larger MoEUT baselines used profiles added to that "
            "file by hand; they are not part of the upstream repository and "
            "are not shipped here. Use one of the names above, or append your "
            "own dict to profiles.py."
        )
    return getattr(profiles, name)
