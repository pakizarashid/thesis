"""
apply_emptytranscript_fallback_patch.py -- fixes Transcriber.__call__ in
src/eval/cloner_watermark_eval.py: its comment already says "Fall back to a
neutral prompt and say so, loudly" when faster-whisper returns an empty
transcript, but the code only prints the warning and then returns the empty
string anyway. That empty ref_text reaches CosyVoice's text_normalize() ->
wetext's token_parser.load(), which does `assert len(input) > 0` and crashes
the whole run mid-way through (not on utterance 0, so the existing seed/config
guard never catches it). MaskGCT requires the same transcript path and will
hit the same crash under the same condition.

Fix: when the transcript comes back empty, substitute a short neutral fallback
sentence (still non-empty) instead of the empty string, so the cloner backend
gets something to normalize instead of crashing. The warning is still printed,
now naming the substituted text.

Usage: python3 apply_emptytranscript_fallback_patch.py [path-to-cloner_watermark_eval.py]
Default path: src/eval/cloner_watermark_eval.py (relative to cwd).
"""
import os
import sys

_DEFAULT_TARGET = "src/eval/cloner_watermark_eval.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

MARKER = "substituting neutral fallback prompt"

OLD = (
    '        if not text:\n'
    '            print(f"[transcribe] WARNING empty transcript for {wav_path}")\n'
    '        return text'
)

NEW = (
    '        if not text:\n'
    '            # 2026-09-15 fix: this used to warn and then return the empty string\n'
    '            # anyway, which crashes CosyVoice/MaskGCT deep inside their own text\n'
    '            # normalizers (wetext\'s token_parser.load does `assert len(input) > 0`)\n'
    '            # -- not on utterance 0, so the seed/config guard never catches it.\n'
    '            # Substitute a short, content-neutral, definitely-non-empty prompt so\n'
    '            # the cloner backend has something to normalize.\n'
    '            text = "This is a voice sample."\n'
    '            print(f"[transcribe] WARNING empty transcript for {wav_path} -- "\n'
    '                  f"substituting neutral fallback prompt {text!r} so the cloner "\n'
    '                  f"backend does not crash on an empty string.")\n'
    '        return text'
)


def _apply(path):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()

    if MARKER in content:
        print(f"[apply_emptytranscript_fallback_patch] {path} already patched -- skipping.")
        return

    count = content.count(OLD)
    if count != 1:
        raise SystemExit(
            f"ABORT: anchor occurs {count} times in {path} (expected 1). "
            f"File may have changed since this patch was written. Nothing written.\n"
            f"--- anchor ---\n{OLD}"
        )

    updated = content.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(updated)

    import ast
    ast.parse(updated)
    print(f"[apply_emptytranscript_fallback_patch] Patched {path} (verified: parses OK).")


if __name__ == "__main__":
    _apply(TARGET)
