"""
apply_xtts_attnmask_shim_patch.py (2026-10-08)

Defensive shim for the XTTS crash seen on 2026-10-08:

    AttributeError: 'NoneType' object has no attribute 'shape'
    (TTS/tts/layers/xtts/gpt_inference.py, GPT2InferenceModel.forward: attention_mask.shape[1])

GPT2InferenceModel.forward() reads attention_mask.shape[1] on every single-token decoding step to compute the position embedding. coqui-tts passes
the mask to generate() explicitly, but some transformers builds do not forward it to forward(), so it arrives as None. XTTS runs batch size 1 with no
padding, so the correct mask is all ones with length (tokens already in the KV cache + tokens fed now). This patch rebuilds exactly that when (and only when)
the mask is None. It does not change behaviour when the mask is supplied.

This is a workaround for a coqui-tts / transformers version mismatch, not a fix of its cause. The clean fix is a fresh session with only
`scripts/setup_env.sh` (transformers>=4.57,<5) and NO `pip install f5-tts` (which pulls transformers 5.x). If XTTS still fails, run the
version cell in the notebook and send its output.

Usage: python patches/apply_xtts_attnmask_shim_patch.py   (idempotent; edits the INSTALLED TTS package)
"""
import importlib.util
import os
import sys

if len(sys.argv) > 1:
    target = sys.argv[1]
else:
    spec = importlib.util.find_spec("TTS")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("[xtts shim] coqui-tts (package 'TTS') is not installed in this environment")
    target = os.path.join(list(spec.submodule_search_locations)[0], "tts", "layers", "xtts", "gpt_inference.py")
src = open(target).read()

MARKER = "THESIS_ATTNMASK_SHIM"
if MARKER in src:
    print(f"[xtts shim] already applied: {target}")
    sys.exit(0)

anchor = "        # Create embedding\n        prefix_len = self.cached_prefix_emb.shape[1]\n"
if src.count(anchor) != 1:
    raise SystemExit(f"[xtts shim] anchor not found exactly once in {target} (this coqui-tts version differs); not patching. "
                     f"Send the file's forward() and the version list.")

shim = anchor + '''        if attention_mask is None:      # THESIS_ATTNMASK_SHIM: batch size 1, no padding -> an all-ones mask is exact
            kv_len = 0
            if past_key_values is not None:
                try:
                    kv_len = int(past_key_values.get_seq_length())
                except AttributeError:
                    kv_len = int(past_key_values[0][0].shape[-2]) if len(past_key_values) else 0
            attention_mask = torch.ones(
                input_ids.shape[0], kv_len + input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
'''
open(target, "w").write(src.replace(anchor, shim))
print(f"[xtts shim] applied: {target}")
