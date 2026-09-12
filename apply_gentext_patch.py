"""
apply_gentext_patch.py

Adds --gen_text_from_transcript to src/eval/cloner_watermark_eval.py (used by
CARRIER-PROBE for F5-TTS/CosyVoice/MaskGCT). Same anchor-based safe-patch pattern
as every other patch script in this project: checks each anchor occurs exactly
once before writing anything; aborts loudly, writing NOTHING, on any mismatch.

WHY: cloner_watermark_eval.py always synthesizes the fixed --gen_text sentence
("This is a test sentence for voice cloning.") regardless of what the reference
utterance actually says. That's fine for ACC-style watermark-detection measurement
(content-invariant), but CARRIER-PROBE compares raw per-layer latents frame-by-frame,
which requires source and clone to contain the SAME words -- otherwise the crash you
just saw (frame-count mismatch) either recurs in a smaller, silently-cropped form, or
the resulting similarity numbers mix "different content" with "different voice" and
aren't interpretable. This patch adds an opt-in flag (default off, so every existing
use of this script for other experiments is unaffected) that makes the clone target
each utterance's own LibriSpeech ground-truth transcript instead.

Usage (run from the repo root, i.e. /kaggle/working/thesis):
    python apply_gentext_patch.py
    # or, to target a different path:
    python apply_gentext_patch.py path/to/cloner_watermark_eval.py
"""
import os
import sys

_DEFAULT_TARGET = "src/eval/cloner_watermark_eval.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

EDITS = [
    (
        '    p.add_argument("--gen_text", type=str,\n'
        '                   default="This is a test sentence for voice cloning.")\n'
        '    p.add_argument("--ref_text", type=str, default="",\n'
        '                   help="Fixed prompt transcript. Leave empty and use --auto_transcribe.")\n',
        '    p.add_argument("--gen_text", type=str,\n'
        '                   default="This is a test sentence for voice cloning.")\n'
        '    p.add_argument("--ref_text", type=str, default="",\n'
        '                   help="Fixed prompt transcript. Leave empty and use --auto_transcribe.")\n'
        '    p.add_argument("--gen_text_from_transcript", action="store_true",\n'
        '                   help="CARRIER-PROBE (2026-09-12): use each utterance\'s own "\n'
        '                        "LibriSpeech ground-truth transcript as BOTH the synthesis "\n'
        '                        "target (gen_text) and the prompt transcript (ref_text), "\n'
        '                        "instead of the fixed --gen_text sentence. Requires "\n'
        '                        "LibriSpeech input (ignored in --input_wav_dir mode, where no "\n'
        '                        "transcript is available). Makes the clone say the same words "\n'
        '                        "as the reference, so a downstream frame-by-frame latent "\n'
        '                        "comparison (carrier_probe.py) is not confounded by content "\n'
        '                        "mismatch. Default off -- every other use of this script for "\n'
        '                        "ACC-style measurement is unaffected.")\n'
    ),
    (
        '    def clone_one(recon_wm, tag):\n'
        '        ref_text = args.ref_text\n'
        '        if transcribe is not None:\n'
        '            ref_text = transcribe(write_ref_wav(recon_wm, args.tmp_dir, tag))\n'
        '        return clone_fn(model, recon_wm, args.gen_text, args.tmp_dir, tag, ref_text, args)\n',
        '    def clone_one(recon_wm, tag, own_transcript=None):\n'
        '        gen_text = args.gen_text\n'
        '        ref_text = args.ref_text\n'
        '        if args.gen_text_from_transcript and own_transcript:\n'
        '            # Known ground truth -- use it for both roles, skip whisper entirely.\n'
        '            gen_text = own_transcript\n'
        '            ref_text = own_transcript\n'
        '        elif transcribe is not None:\n'
        '            ref_text = transcribe(write_ref_wav(recon_wm, args.tmp_dir, tag))\n'
        '        return clone_fn(model, recon_wm, gen_text, args.tmp_dir, tag, ref_text, args)\n'
    ),
    (
        '        batch = next(iter(loader))\n'
        '        clean = batch["waveform"].to(device)\n'
        '        msg = random_message(16, 1, device, seed=args.message_seed_base)\n'
        '        with torch.no_grad():\n'
        '            recon_wm = backbone.forward_full(clean, msg)["recon_wm"]\n'
        '        cloned = clone_one(recon_wm, "diag")\n',
        '        batch = next(iter(loader))\n'
        '        clean = batch["waveform"].to(device)\n'
        '        msg = random_message(16, 1, device, seed=args.message_seed_base)\n'
        '        with torch.no_grad():\n'
        '            recon_wm = backbone.forward_full(clean, msg)["recon_wm"]\n'
        '        diag_transcript = (batch.get("transcript") or [None])[0]\n'
        '        cloned = clone_one(recon_wm, "diag", own_transcript=diag_transcript)\n',
    ),
    (
        '        else:\n'
        '            clean = item["waveform"].to(device)\n'
        '            with torch.no_grad():\n'
        '                recon_wm = backbone.forward_full(clean, msg)["recon_wm"]\n'
        '\n'
        '        cloned = clone_one(recon_wm, f"u{i}")\n',
        '        else:\n'
        '            clean = item["waveform"].to(device)\n'
        '            with torch.no_grad():\n'
        '                recon_wm = backbone.forward_full(clean, msg)["recon_wm"]\n'
        '\n'
        '        loop_transcript = (item.get("transcript") or [None])[0] if kind == "batch" else None\n'
        '        cloned = clone_one(recon_wm, f"u{i}", own_transcript=loop_transcript)\n',
    ),
]


def _apply(path, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}). "
                          f"Pass the path explicitly: python apply_gentext_patch.py <path>")
    content = open(path, "r").read()

    # IDEMPOTENCY: /kaggle/working persists across kernel restarts even though pip
    # installs don't, so it's easy to end up running this a second time against a
    # file it already patched in an earlier session. Detect that up front and exit
    # cleanly instead of aborting on edit 1's anchor (already replaced by edit 0's
    # own output) with an alarming-looking "occurs 0 times" message.
    if "--gen_text_from_transcript" in content:
        print(f"[apply_gentext_patch] {path} already has --gen_text_from_transcript -- "
              f"nothing to do (this is expected if you've run this patch before in an "
              f"earlier Kaggle session; /kaggle/working persists across restarts even "
              f"though pip installs don't). Skipping.")
        return

    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            raise SystemExit(
                f"ABORT: edit {i} anchor occurs {count} times in {path} (expected 1). "
                f"Nothing has been written -- the file is untouched. "
                f"Anchor (first 200 chars):\n{old[:200]!r}"
            )
        new_content = new_content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[apply_gentext_patch] Patched {path} ({len(edits)} edits applied).")


if __name__ == "__main__":
    _apply(TARGET, EDITS)
