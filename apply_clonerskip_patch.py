"""
apply_clonerskip_patch.py -- makes the per-utterance loop in
src/eval/cloner_watermark_eval.py tolerate a cloner backend crashing on a
single utterance, instead of taking the whole unattended n=100 run down with
it.

WHY THIS EXISTS
---------------
apply_emptytranscript_fallback_patch.py already fixed the case where
faster-whisper returns a literally-empty transcript (CosyVoice's wetext
normalizer asserts len(input) > 0 on that). But wetext can ALSO raise the
same AssertionError deeper inside its own reordering logic on a transcript
that is NOT empty -- e.g. a short, punctuation-heavy, or otherwise unusual
whisper transcription that its span-reordering step reduces to zero tokens
internally. There is no tractable way to pre-filter every string wetext
chokes on; the fallback-text patch cannot catch this because the crashing
text isn't empty going in.

FIX
---
Wrap only the clone_fn call (clone_one(...)) in a try/except. On failure,
print a loud, specific warning naming the utterance index and the exception,
then `continue` to the next utterance instead of crashing the process. This
mirrors the project's existing "n_completed can be less than n_utterances"
convention (see _dump(): n_completed = len(accs_clone)) -- a run that skips
a handful of pathological utterances out of 100 still produces a valid,
honestly-labeled result; a run that dies at utterance 47 produces nothing.

This is deliberately a broad except Exception, not a narrow one, because the
failure modes here are inside third-party cloner code (CosyVoice/MaskGCT/
wetext internals) that this project does not control and cannot enumerate in
advance. The tradeoff: a genuine bug in OUR code at this call site would also
get silently skipped and misread as "just a flaky utterance." Mitigated by
printing the full exception type+message every time (grep the log for
"SKIPPED" afterward) -- if the skip count for a run is more than a couple
out of 100, treat that as a signal to investigate rather than accept the
result at face value.

Usage: python3 apply_clonerskip_patch.py [path-to-cloner_watermark_eval.py]
Default path: src/eval/cloner_watermark_eval.py (relative to cwd).
"""
import os
import sys

_DEFAULT_TARGET = "src/eval/cloner_watermark_eval.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

MARKER = "SKIPPED -- {args.cloner} cloning failed"

OLD = (
    '        loop_transcript = (item.get("transcript") or [None])[0] if kind == "batch" else None\n'
    '        cloned = clone_one(recon_wm, f"u{i}", own_transcript=loop_transcript)\n'
    '\n'
    '        a_src = detect_acc(backbone, recon_wm, msg)'
)

NEW = (
    '        loop_transcript = (item.get("transcript") or [None])[0] if kind == "batch" else None\n'
    '        # 2026-09-15 fix: a cloner backend (CosyVoice/MaskGCT, via wetext\'s text\n'
    '        # normalizer) can raise on a single pathological utterance -- not just a\n'
    '        # literally-empty transcript (handled separately in Transcriber.__call__),\n'
    '        # but also a short/unusual one that its OWN internal reordering reduces to\n'
    '        # zero tokens. Rather than enumerate every string a third-party normalizer\n'
    '        # chokes on, skip this utterance loudly and keep the unattended run alive;\n'
    '        # n_completed (see _dump) already tolerates fewer results than n_utterances.\n'
    '        try:\n'
    '            cloned = clone_one(recon_wm, f"u{i}", own_transcript=loop_transcript)\n'
    '        except Exception as e:\n'
    '            print(f"  [{i}] SKIPPED -- {args.cloner} cloning failed: "\n'
    '                  f"{type(e).__name__}: {e}", flush=True)\n'
    '            continue\n'
    '\n'
    '        a_src = detect_acc(backbone, recon_wm, msg)'
)


def _apply(path):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()

    if MARKER in content:
        print(f"[apply_clonerskip_patch] {path} already patched -- skipping.")
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
    print(f"[apply_clonerskip_patch] Patched {path} (verified: parses OK).")


if __name__ == "__main__":
    _apply(TARGET)
