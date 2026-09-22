"""
apply_cosyvoice_promptfix_patch.py

Fixes clone_cosyvoice() in src/eval/cloner_watermark_eval.py: this installed
CosyVoice version's frontend_zero_shot -> _extract_speech_feat calls
load_wav(prompt_wav, 24000) ITSELF on whatever is passed to inference_zero_shot.
Pre-loading to a 16kHz tensor first (the old "try tensor, fall back to path" logic)
makes that internal call try to torchaudio-decode an already-loaded float tensor,
which crashes with "video_tensor must be kUInt8" -- not caught by the existing
try/except, since that only wraps the OUTER load_wav call, not the one inside
inference_zero_shot. Fix: always pass the path; the frontend loads it itself.
"""
import os, sys

_DEFAULT_TARGET = "src/eval/cloner_watermark_eval.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

OLD = (
    '    prompt = None\n'
    '    try:\n'
    '        from cosyvoice.utils.file_utils import load_wav\n'
    '        prompt = load_wav(ref_path, 16000)\n'
    '    except Exception:\n'
    '        prompt = ref_path\n'
)
NEW = (
    '    # 2026-09-14 fix: this installed CosyVoice frontend calls load_wav(prompt_wav,\n'
    '    # 24000) on whatever is passed here -- a pre-loaded tensor breaks that internal\n'
    '    # call ("video_tensor must be kUInt8"). Always pass the path; the frontend\n'
    '    # loads it itself at the model\'s own sample rate.\n'
    '    prompt = ref_path\n'
)

def _apply(path, old, new):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if new.strip() in content:
        print(f"[apply_cosyvoice_promptfix_patch] {path} already patched -- skipping.")
        return
    count = content.count(old)
    if count != 1:
        raise SystemExit(f"ABORT: anchor occurs {count} times in {path} (expected 1). Nothing written.\n{old!r}")
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    print(f"[apply_cosyvoice_promptfix_patch] Patched {path}.")

if __name__ == "__main__":
    _apply(TARGET, OLD, NEW)
