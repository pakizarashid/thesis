"""
apply_xtts_backend_patch.py (2026-10-08)

Adds XTTS-v2 as a fourth backend of src/eval/cloner_watermark_eval.py (--cloner xtts) and makes the script record WHICH utterance
indices completed ("completed_indices" in the result JSON).

WHY
---
xtts_transfer_eval.py re-runs PGD itself, cannot read pre-made WAVs and saves no clones, so XTTS could never be scored on the SAME
protected clips as F5-TTS / MaskGCT / CosyVoice (no per-clip ECAPA, no attacker-wins, no clone WER). With this backend XTTS goes through the
identical pre-made-audio path: --input_wav_dir <dir> --wav_suffix {unprotected,protected} --save_clones_dir <dir>.

The clone call is the same coqui-tts high-level API xtts_transfer_eval.py already uses (tts_to_file, language="en", the same fixed gen_text),
so XTTS numbers from the two scripts are comparable. XTTS does not need a transcript, so --auto_transcribe is not required.

completed_indices: a backend that crashes on one utterance is skipped (SKIPPED --), so acc_clone_values is shorter than 100 and its i-th entry is NOT
clip i. The index list makes per-clip joins (ACC with ECAPA SIM) exact instead of positional.

Usage: python patches/apply_xtts_backend_patch.py [path-to-cloner_watermark_eval.py]   (idempotent)
"""
import os
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "src/eval/cloner_watermark_eval.py"
src = open(TARGET).read()
orig = src


def sub(old, new, label):
    global src
    if src.count(old) != 1:
        raise SystemExit(f"[xtts-backend patch] cannot apply '{label}': expected exactly 1 match, found {src.count(old)} in {TARGET}")
    src = src.replace(old, new)
    print(f"[xtts-backend patch] applied: {label}")


XTTS_BLOCK = '''# --------------------------------------------------------------------------------
# backend: XTTS-v2  (GPT audio-prompt tokens; measured 0.6119 by xtts_transfer_eval.py)
# --------------------------------------------------------------------------------

def load_xtts(device, args):
    # Kaggle cells are non-interactive: coqui-tts asks to accept the non-commercial CPML licence with input() and
    # raises EOFError otherwise. This env var is its documented way to auto-accept (academic / non-commercial use).
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    from TTS.api import TTS
    print("[xtts] loading tts_models/multilingual/multi-dataset/xtts_v2 (first run downloads weights)...")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def clone_xtts(model, speaker_audio_16k, gen_text, tmp_dir, tag, ref_text, args):
    import torchaudio
    ref_path = write_ref_wav(speaker_audio_16k, tmp_dir, tag)
    out_path = os.path.join(tmp_dir, f"out_{tag}.wav")
    model.tts_to_file(text=gen_text, speaker_wav=ref_path, language="en", file_path=out_path)
    wav, sr = torchaudio.load(out_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return to_16k_tensor(wav, sr, speaker_audio_16k.device)


_DONE_IDX = []          # indices of the utterances that completed (see _dump)


'''

if "def clone_xtts" not in src:
    sub("BACKENDS = {\n", XTTS_BLOCK + "BACKENDS = {\n", "xtts functions")
    sub('    "maskgct":   (load_maskgct,   clone_maskgct,   "masked RVQ acoustic infilling", 0.957),\n}',
        '    "maskgct":   (load_maskgct,   clone_maskgct,   "masked RVQ acoustic infilling", 0.957),\n'
        '    "xtts":      (load_xtts,      clone_xtts,      "GPT audio-prompt tokens",       None),\n}',
        "BACKENDS entry")
else:
    print("[xtts-backend patch] xtts backend already present")

if "_DONE_IDX.append" not in src:
    if "_DONE_IDX = []" not in src:       # (backend block already inserted it in the normal path)
        sub("BACKENDS = {\n", "_DONE_IDX = []          # indices of the utterances that completed (see _dump)\n\nBACKENDS = {\n", "_DONE_IDX definition")
    sub("        accs_clone.append(a_cln)\n", "        accs_clone.append(a_cln)\n        _DONE_IDX.append(i)\n", "record completed index")
    sub('                "acc_clone_values": accs_clone,\n',
        '                "acc_clone_values": accs_clone,\n                "completed_indices": list(_DONE_IDX),\n', "dump completed_indices")
else:
    print("[xtts-backend patch] completed_indices already present")

if src != orig:
    open(TARGET, "w").write(src)
    print(f"[xtts-backend patch] wrote {TARGET}")
else:
    print("[xtts-backend patch] nothing to do")
