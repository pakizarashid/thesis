"""
scripts/patch_audiopure.py

The AudioPure submodule (external/audiopure) is an unmaintained repo built
against torchaudio==0.11.0. Its dataset.py imports download_url/extract_archive
from torchaudio.datasets.utils, neither of which exist in current torchaudio --
this breaks importing diffwave_ddpm.py at all, even though we never call the
functions that actually use them (we only need the DiffWave model class and
create_diffwave_model()).

IMPORTANT: these are local edits to a git SUBMODULE. `git submodule update`
always re-fetches AudioPure's ORIGINAL upstream files -- our fix does NOT
persist through that automatically. Rather than fork AudioPure for two lines,
this script re-applies the patch every session. It's idempotent (safe to run
whether the file is already patched, unpatched, or partially patched from an
earlier manual attempt) -- run it once per fresh submodule checkout.

Usage:
    python scripts/patch_audiopure.py
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIOPURE_ROOT = os.path.join(REPO_ROOT, "external", "audiopure")


def patch_dataset_import():
    path = os.path.join(AUDIOPURE_ROOT, "diffusion_models", "DiffWave_Unconditional", "dataset.py")
    if not os.path.exists(path):
        print(f"[patch_audiopure] SKIP - {path} not found (submodule not checked out yet?)")
        return

    with open(path) as f:
        content = f.read()

    # Matches either the original multi-line import block, or an earlier
    # single-line partial-patch attempt -- idempotent regardless of which
    # state the file is currently in.
    pattern = re.compile(
        r"from torchaudio\.datasets\.utils import \([^)]*\)|"
        r"from torchaudio\.datasets\.utils import extract_archive"
    )
    replacement = (
        "# torchaudio.datasets.utils import removed by patch_audiopure.py: "
        "download_url/extract_archive don't exist in current torchaudio, "
        "and aren't used by anything this project calls (only by "
        "load_Qualcomm_keyword, which we never invoke)"
    )

    new_content, n = pattern.subn(replacement, content)
    if n > 0:
        with open(path, "w") as f:
            f.write(new_content)
        print(f"[patch_audiopure] Patched {path} ({n} match(es) replaced)")
    elif replacement in content:
        print(f"[patch_audiopure] Already patched: {path}")
    else:
        print(f"[patch_audiopure] WARNING - no match found in {path}. "
              f"The file may have changed upstream -- inspect manually.")


def ensure_init_files():
    needed = [
        os.path.join(AUDIOPURE_ROOT, "__init__.py"),
        os.path.join(AUDIOPURE_ROOT, "diffusion_models", "__init__.py"),
        os.path.join(AUDIOPURE_ROOT, "diffusion_models", "DiffWave_Unconditional", "__init__.py"),
    ]
    for path in needed:
        if not os.path.exists(os.path.dirname(path)):
            print(f"[patch_audiopure] SKIP - directory for {path} doesn't exist (submodule not checked out?)")
            continue
        if not os.path.exists(path):
            open(path, "a").close()
            print(f"[patch_audiopure] Created {path}")
        else:
            print(f"[patch_audiopure] Already exists: {path}")


if __name__ == "__main__":
    print("[patch_audiopure] Patching AudioPure submodule for compatibility with current torchaudio...")
    ensure_init_files()
    patch_dataset_import()
    print("[patch_audiopure] Done. Verify with: "
          "python -c \"import sys; sys.path.insert(0, '.'); "
          "from external.audiopure.diffusion_models.diffwave_ddpm import create_diffwave_model; "
          "print('Import OK')\"")
