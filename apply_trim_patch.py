"""
apply_trim_patch.py

Adds the same trailing-silence trim to src/eval/cloner_watermark_eval.py that
gen_samples_yourtts.py/gen_samples_xtts.py already got -- F5-TTS/CosyVoice/MaskGCT
go through this file with --crop_seconds 20 too, so without this they'd hit the
exact same "padding read as content mismatch" bug.

Anchors are deliberately taken from TEXT ALREADY CONFIRMED LIVE by
probe_clone_one.py's output (the clone_one() function, byte-exact) for edit 0, so
that one is safe. Edits 1/2 (the two `clean = ...to(device)` lines) are NOT
independently reconfirmed this round -- they're unchanged by every patch so far,
so very likely still exactly as originally read, but two prior misses on this file
earn some caution. So: same "abort, write nothing" safety as always, PLUS if 1/2
fail, this prints nearby context automatically (grep-style) instead of just the
raw anchor -- so a failure here is self-diagnosing in the same message, no need
for a separate probe round-trip.
"""
import os
import re
import sys

_DEFAULT_TARGET = "src/eval/cloner_watermark_eval.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

TRIM_FN = (
    'def trim_trailing_silence(waveform, eps: float = 1e-4):\n'
    '        """\n'
    '        CARRIER-PROBE (2026-09-12, third fix). Cuts trailing near-zero samples --\n'
    '        the _crop_or_pad padding LibriSpeechSubset appends when an utterance is\n'
    '        shorter than --crop_seconds -- off a reference waveform. Without this,\n'
    '        --crop_seconds 20 pads most (shorter) LibriSpeech utterances with several\n'
    '        seconds of trailing silence, which carrier_probe.py\'s frame-count-mismatch\n'
    '        check misreads as a content mismatch. Same fix already applied to\n'
    '        gen_samples_yourtts.py/gen_samples_xtts.py -- see that file\'s docstring.\n'
    '        """\n'
    '        w = waveform.squeeze(0) if waveform.dim() > 1 else waveform\n'
    '        nz = (w.abs() > eps).nonzero()\n'
    '        if nz.numel() == 0:\n'
    '            return waveform\n'
    '        last = nz[-1].item()\n'
    '        trimmed = w[: last + 1]\n'
    '        return trimmed.unsqueeze(0) if waveform.dim() > 1 else trimmed\n'
    '\n'
    '    '
)

EDITS = [
    (
        # CONFIRMED byte-exact by probe_clone_one.py's output.
        'def clone_one(recon_wm, tag, own_transcript=None):\n'
        '        gen_text = args.gen_text\n',
        TRIM_FN + 'def clone_one(recon_wm, tag, own_transcript=None):\n'
        '        gen_text = args.gen_text\n',
    ),
    (
        # NOT reconfirmed this round -- unchanged by every patch so far, so very
        # likely intact, but see the fallback search below if this misses.
        '        batch = next(iter(loader))\n'
        '        clean = batch["waveform"].to(device)\n'
        '        msg = random_message(16, 1, device, seed=args.message_seed_base)\n',
        '        batch = next(iter(loader))\n'
        '        clean = trim_trailing_silence(batch["waveform"][0]).unsqueeze(0).to(device)\n'
        '        msg = random_message(16, 1, device, seed=args.message_seed_base)\n',
    ),
    (
        '        else:\n'
        '            clean = item["waveform"].to(device)\n'
        '            with torch.no_grad():\n',
        '        else:\n'
        '            clean = trim_trailing_silence(item["waveform"][0]).unsqueeze(0).to(device)\n'
        '            with torch.no_grad():\n',
    ),
]


def _nearby(content, fragment, width=80):
    hits = []
    for m in re.finditer(re.escape(fragment), content):
        start = max(0, m.start() - width)
        end = min(len(content), m.end() + width)
        hits.append(content[start:end])
    return hits


def _apply(path, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()

    if "def trim_trailing_silence(" in content:
        print(f"[apply_trim_patch] {path} already has trim_trailing_silence -- "
              f"nothing to do. Skipping.")
        return

    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            print(f"ABORT: edit {i} anchor occurs {count} times in {path} (expected 1). "
                  f"Nothing has been written -- the file is untouched.")
            # Self-diagnose: search for the most distinctive short fragment from this
            # anchor and print what's actually around it, so a failure here doesn't
            # need a separate probe round-trip.
            probe_fragments = {
                0: 'def clone_one(',
                1: 'clean = batch["waveform"]',
                2: 'clean = item["waveform"]',
            }
            frag = probe_fragments.get(i)
            if frag:
                hits = _nearby(content, frag)
                if hits:
                    print(f"\nContext around {frag!r} as it actually appears in {path}:")
                    for h in hits:
                        print(f"  ...{h!r}...")
                else:
                    print(f"\n{frag!r} does not appear in {path} at all.")
            raise SystemExit(1)
        new_content = new_content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[apply_trim_patch] Patched {path} ({len(edits)} edits applied).")


if __name__ == "__main__":
    _apply(TARGET, EDITS)
