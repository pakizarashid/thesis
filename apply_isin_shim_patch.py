"""
apply_isin_shim_patch.py

Fixes: ImportError: cannot import name 'isin_mps_friendly' from 'transformers.pytorch_utils'

Root cause (confirmed via upstream GitHub issues, not guessed):
  - transformers >= 5.1 removed `isin_mps_friendly` from transformers/pytorch_utils.py.
  - Coqui TTS's tortoise/autoregressive.py (loaded as part of importing XttsConfig/GPT,
    which happens as soon as anything does `from TTS...`) still does
    `from transformers.pytorch_utils import isin_mps_friendly as isin` at import time.
  - src/models/surrogate_vc.py does `from TTS.tts.utils.helpers import generate_path,
    sequence_mask`, which triggers TTS's package __init__ chain -> XttsConfig -> GPT ->
    tortoise/autoregressive.py -> the broken import.
  - This almost certainly started failing right after the F5-TTS/Whisper cell ran, because
    that cell's own dependency resolution silently pulled in transformers>=5.1 (needed by
    newer Whisper/F5-TTS code), which is incompatible with Coqui TTS's vendored tortoise code.
  - Upstream fix status (idiap/coqui-ai-TTS#558): open PR, no released fix yet. Recommended
    workaround there is pinning transformers<5.1 -- but that risks breaking whatever the
    F5-TTS/Whisper cell needs from the newer transformers, and re-installing packages on
    Kaggle mid-session has repeatedly caused this project's dependency-fragmentation bugs
    before. So instead of touching the installed package, this patch restores the missing
    name directly: it defines a tiny compatible shim (isin_mps_friendly on CUDA/CPU is just
    torch.isin -- the MPS-specific fallback path in the original doesn't apply on Kaggle's
    CUDA runtime) and injects it into transformers.pytorch_utils BEFORE anything imports TTS.

This makes the fix work regardless of which transformers version ends up installed, and
survives future pip installs in the same notebook without needing a kernel restart.

Usage: python3 apply_isin_shim_patch.py <path-to-repo-root>
"""
import sys
import re

MARKER = "# --- isin_mps_friendly shim (transformers>=5.1 removed it; TTS's tortoise code still imports it) ---"


def _apply(path, edits):
    with open(path, "r") as f:
        content = f.read()

    if MARKER in content:
        print(f"[SKIP] {path} already patched")
        return

    for old, new in edits:
        count = content.count(old)
        if count != 1:
            print(f"[ABORT] {path}: expected 1 match for anchor, found {count}")
            print("----- anchor -----")
            print(old)
            sys.exit(1)
        content = content.replace(old, new)

    with open(path, "w") as f:
        f.write(content)
    print(f"[OK] patched {path}")


def patch_surrogate_vc(repo_root):
    path = f"{repo_root}/src/models/surrogate_vc.py"
    old = (
        "import os\n"
        "import sys\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "from TTS.tts.utils.helpers import generate_path, sequence_mask\n"
    )
    new = (
        "import os\n"
        "import sys\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        f"{MARKER}\n"
        "import transformers.pytorch_utils as _ptu\n"
        "if not hasattr(_ptu, \"isin_mps_friendly\"):\n"
        "    def _isin_mps_friendly_shim(elements, test_elements):\n"
        "        # Original only special-cased MPS devices on old torch; on CUDA/CPU\n"
        "        # (this project's runtime) it was always just torch.isin.\n"
        "        return torch.isin(elements, test_elements)\n"
        "    _ptu.isin_mps_friendly = _isin_mps_friendly_shim\n"
        "# --- end isin_mps_friendly shim ---\n"
        "\n"
        "from TTS.tts.utils.helpers import generate_path, sequence_mask\n"
    )
    _apply(path, [(old, new)])


if __name__ == "__main__":
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."
    patch_surrogate_vc(repo_root)
    print("Done.")
