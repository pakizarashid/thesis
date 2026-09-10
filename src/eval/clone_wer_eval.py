"""
src/eval/clone_wer_eval.py

Completes the like-for-like table against SafeSpeech by adding the metric it reports
alongside SIM: **WER of the synthesised (cloned) speech**.

WHY THIS IS NEEDED
------------------
SafeSpeech's Table 1 reports BOTH metrics, and they measure different things:

    SafeSpeech (LibriTTS, BERT-VITS2)     clean      protected
      WER                                 24.024%    99.610%
      SIM                                  0.604      0.204

WER captures whether the attacker's clone is still *intelligible* -- a clone can have low
speaker similarity yet remain usable, or be so degraded that it is worthless regardless of
timbre. Reporting SIM alone tells half the story. Their DEMUCS row (WER 99.610% -> 57.329%)
is a WER claim, so any comparison to it requires this number.

Higher WER = more disrupted = better protection.

DESIGN: STANDALONE BY DESIGN
-----------------------------
Imports nothing from this project's pipeline -- no backbone, no surrogate, no VoiceMark
code. Same isolation principle as ecapa_sim_eval.py, and for the same reason: installing
`denoiser` once downgraded omegaconf and broke checkpoint loading for every script that
touched the backbone. A scoring script that only reads WAVs cannot cause that.

    pip install openai-whisper jiwer

INPUT
-----
The WAV layout written by demucs_fallback_eval.py --save_clones_dir:

    <dir>/sample{i}_clone_clean.wav      clone of unprotected audio
    <dir>/sample{i}_clone_protected.wav  clone of protected audio
    <dir>/sample{i}_clone_demucs.wav     clone after the attacker denoised

Every condition present is scored separately. `sample{i}_reference.wav` is SKIPPED: it is
real LibriSpeech speech whose transcript is not the synthesis prompt, so WER against
--reference_text would be meaningless for it.

TEXT NORMALISATION -- carried over from quality_metrics.py's documented finding
-------------------------------------------------------------------------------
Normalises manually (lowercase, strip punctuation, collapse whitespace) and calls
jiwer.wer(reference, hypothesis) positionally, rather than using truth_transform /
reference_transform. Two reasons, both real: jiwer 3.0+ renamed truth->reference (breaking
across versions), and jiwer 3.0.0 has a confirmed correctness bug where
reference_transform yields wrong WER values (jitsi/jiwer issue #76). Manual normalisation
sidesteps both and is stable across versions.

Usage:
    python src/eval/clone_wer_eval.py \\
        --sample_dir ./audio_samples/demucs_speech \\
        --output results/results_clone_wer_speech.json
"""

import os
import re
import glob
import json
import argparse


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--reference_text", type=str,
                   default="This is a test sentence for voice cloning.",
                   help="MUST match --surrogate_text used when the clones were generated, "
                        "or every WER is nonsense.")
    p.add_argument("--whisper_model_size", type=str, default="base",
                   help="'base' is the speed/accuracy tradeoff used elsewhere in this repo. "
                        "Whichever is used, keep it constant across conditions -- comparing "
                        "WER from different ASR models is invalid.")
    p.add_argument("--skip_conditions", type=str, default="reference",
                   help="Comma-separated condition names to skip (their transcript is not "
                        "--reference_text).")
    args = p.parse_args()

    import whisper
    import jiwer

    skip = {s.strip() for s in args.skip_conditions.split(",") if s.strip()}

    files = sorted(glob.glob(os.path.join(args.sample_dir, "sample*_*.wav")))
    if not files:
        raise SystemExit(f"No sample*_*.wav in {args.sample_dir}. Generate them first with "
                         f"demucs_fallback_eval.py --save_clones_dir ...")

    print(f"[wer] loading Whisper ({args.whisper_model_size})...")
    model = whisper.load_model(args.whisper_model_size)
    ref_norm = normalize(args.reference_text)
    print(f"[wer] reference text: \"{args.reference_text}\"")
    print(f"[wer] {len(files)} wav files found in {args.sample_dir}\n")

    per_condition = {}
    transcripts = {}
    for path in files:
        base = os.path.basename(path)
        m = re.match(r"sample(\d+)_(.+)\.wav$", base)
        if not m:
            continue
        idx, cond = m.group(1), m.group(2)
        if cond in skip:
            continue
        res = model.transcribe(path, language="en")
        hyp = res["text"].strip()
        wer = jiwer.wer(ref_norm, normalize(hyp))
        per_condition.setdefault(cond, []).append(wer)
        transcripts.setdefault(cond, []).append(hyp)

    print(f"{'=' * 84}")
    print(f"CLONE WER  (higher = more disrupted = better protection)")
    print(f"{'=' * 84}")
    print(f"{'condition':<26}{'mean WER':>12}{'median':>10}{'n':>6}")
    results = {}
    for cond, vals in per_condition.items():
        v = sorted(vals)
        n = len(v)
        mean = sum(v) / n
        median = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
        results[cond] = {"mean_wer": mean, "median_wer": median, "n": n,
                         "values": vals, "transcripts": transcripts[cond]}
        print(f"{cond:<26}{mean * 100:>11.2f}%{median * 100:>9.2f}%{n:>6}")

    print(f"\nSafeSpeech reference points (LibriTTS, BERT-VITS2, fine-tuning threat model):")
    print(f"  clean 24.024%   protected 99.610%   after DEMUCS 57.329%   after AudioPure 85.711%")
    print(f"\nCOMPARISON CAVEAT: their WER comes from a fine-tuned BERT-VITS2 and a different")
    print(f"corpus; this is zero-shot YourTTS on LibriSpeech. As with SIM, the fair reading is")
    print(f"the RELATIVE change (clean -> protected -> after attack), not the absolute value.")
    print(f"WER is also unbounded above 100% (insertions), so cap interpretation accordingly.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"label": "clone_wer", "sample_dir": args.sample_dir,
                       "reference_text": args.reference_text,
                       "whisper_model": args.whisper_model_size,
                       "safespeech_reference": {"clean": 0.24024, "protected": 0.99610,
                                                 "after_demucs": 0.57329,
                                                 "after_audiopure": 0.85711},
                       "results": results}, f, indent=2)
        print(f"\n[main] Saved to {args.output}")


if __name__ == "__main__":
    main()
