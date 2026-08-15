"""
src/eval/quality_metrics.py

Standard objective quality metrics, not yet used anywhere in this project --
added in direct response to the gap: no PESQ, STOI, or WER had been computed.

Two genuinely different measurements, requiring different validity conditions:

1. PESQ + STOI (clean vs watermarked): valid because content is IDENTICAL and
   frame-aligned. Answers the open question from STAGE1_WRITEUP.md Section 10
   (waveform correlation was ~0.83, unclear if that means audible distortion)
   with actual standard perceptual-quality metrics instead of a raw
   correlation number.

2. WER (cloned audio vs the known reference text fed to the surrogate):
   valid because we KNOW the ground-truth text ("This is a test sentence for
   voice cloning.") -- the surrogate was told to synthesize exactly that.
   Transcribing the cloned output and comparing against that known text
   measures intelligibility, independent of speaker identity (which SIM
   already measures). This is the SAME metric pairing (WER + SIM) SafeSpeech's
   own paper reports -- we had SIM, this adds WER.

PESQ/STOI are NOT computed on cloned audio, since clean and cloned audio have
different content/duration/alignment -- neither metric is defined for that
comparison (see audio_diff_analysis.py's existing explanation of the same
constraint).

Usage:
    python src/eval/quality_metrics.py --sample_dir ./audio_samples/baseline
    python src/eval/quality_metrics.py --sample_dir ./audio_samples/stage2_sim_longrun
"""

import os
import argparse
import json
import numpy as np
import soundfile as sf
import librosa


def compute_si_snr(clean_wav, watermarked_wav) -> float:
    """
    Scale-invariant SNR (SI-SNR / SI-SDR), the standard source-separation
    metric (Le Roux et al.) -- crucially, this is the EXACT metric VoiceMark's
    own paper reports (their Table 3: PESQ 2.20, SI-SNR 2.01 dB, STOI 0.89),
    not the plain SNR computed above. Implemented specifically to make a true
    apples-to-apples comparison against their published numbers, rather than
    the approximate one plain SNR gave.

    Projects the watermarked signal onto the clean signal (removing any pure
    scale/amplitude mismatch) before computing the ratio -- this is what
    makes it "scale-invariant" and distinct from plain SNR above.
    """
    min_len = min(len(clean_wav), len(watermarked_wav))
    s = clean_wav[:min_len].astype(np.float64)
    s_hat = watermarked_wav[:min_len].astype(np.float64)

    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)

    s_target = (np.dot(s_hat, s) / (np.dot(s, s) + 1e-8)) * s
    e_noise = s_hat - s_target

    si_snr_db = 10 * np.log10((np.sum(s_target ** 2) + 1e-8) / (np.sum(e_noise ** 2) + 1e-8))
    return si_snr_db


def compute_snr(clean_wav, watermarked_wav) -> float:
    """
    Signal-to-noise ratio, in dB, between the original clean audio (the
    'signal') and the watermark/disruption perturbation (the 'noise' = the
    difference between watermarked and clean). This is the standard way
    SafeSpeech and similar adversarial-perturbation work report HOW STRONG
    the perturbation is -- distinct from PESQ/STOI, which measure perceptual
    QUALITY impact rather than raw perturbation magnitude.

    Higher SNR = smaller/more imperceptible perturbation.
    Lower SNR = larger/more aggressive perturbation.

    Requires clean_wav and watermarked_wav to be the SAME content, frame-
    aligned (same validity constraint as PESQ/STOI -- not meaningful for
    clean-vs-cloned, which have different content/duration).
    """
    min_len = min(len(clean_wav), len(watermarked_wav))
    clean = clean_wav[:min_len].astype(np.float64)
    watermarked = watermarked_wav[:min_len].astype(np.float64)

    noise = watermarked - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")  # watermarked is bit-identical to clean
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def compute_pesq_stoi(clean_wav, watermarked_wav, sr):
    """
    PESQ requires 8kHz or 16kHz input (our audio is already 16kHz, no resample
    needed). Returns (pesq_score, stoi_score). PESQ range: roughly -0.5 (bad)
    to 4.5 (identical). STOI range: 0 (unintelligible) to 1 (identical).
    """
    from pesq import pesq
    from pystoi import stoi

    min_len = min(len(clean_wav), len(watermarked_wav))
    clean_wav = clean_wav[:min_len].astype(np.float64)
    watermarked_wav = watermarked_wav[:min_len].astype(np.float64)

    pesq_score = pesq(sr, clean_wav, watermarked_wav, "wb")  # wideband mode, correct for 16kHz
    stoi_score = stoi(clean_wav, watermarked_wav, sr, extended=False)
    return pesq_score, stoi_score


def compute_wer(audio_path, reference_text, whisper_model):
    """
    Transcribes audio_path via Whisper, computes word error rate against
    reference_text. Requires: pip install openai-whisper jiwer

    NOTE: normalizes text manually (lowercase, strip punctuation/whitespace)
    with plain Python BEFORE calling jiwer.wer(), rather than using jiwer's
    truth_transform/reference_transform kwargs. Two real problems with those
    kwargs: (1) jiwer 3.0+ renamed truth/truth_transform to
    reference/reference_transform, breaking on older/newer installs
    depending on version, and (2) a documented correctness bug in jiwer
    3.0.0 specifically, where using reference_transform produces WRONG WER
    values compared to the old truth_transform name (confirmed via
    jitsi/jiwer GitHub issue #76). Manual normalization + jiwer's plain
    two-argument wer(reference, hypothesis) form sidesteps both issues and
    is stable across jiwer versions.
    """
    import re
    import jiwer

    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # strip punctuation
        text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
        return text

    result = whisper_model.transcribe(audio_path, language="en")
    hypothesis = result["text"].strip()

    wer_score = jiwer.wer(normalize(reference_text), normalize(hypothesis))
    return wer_score, hypothesis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", type=str, required=True,
                    help="Directory with sampleN_clean.wav / sampleN_watermarked.wav / sampleN_cloned.wav "
                         "(from save_audio_samples.py)")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--reference_text", type=str, default="This is a test sentence for voice cloning.",
                    help="Must match --surrogate_text used when generating the samples.")
    p.add_argument("--whisper_model_size", type=str, default="base",
                    help="Whisper model size -- 'base' is a reasonable speed/accuracy tradeoff for this check.")
    p.add_argument("--skip_wer", action="store_true", help="Skip WER (whisper download/load can be slow).")
    p.add_argument("--output", type=str, default=None,
                    help="Save results as JSON (recommended -- without this, results only exist in "
                         "printed stdout and are lost if not manually captured).")
    args = p.parse_args()

    print(f"\n{'=' * 60}\nQuality metrics for: {args.sample_dir}\n{'=' * 60}")

    results_out = {"sample_dir": args.sample_dir}

    # --- SNR / PESQ / STOI on clean vs watermarked ---
    pesq_scores, stoi_scores, snr_scores, si_snr_scores = [], [], [], []
    for i in range(args.n_samples):
        clean_path = os.path.join(args.sample_dir, f"sample{i}_clean.wav")
        wm_path = os.path.join(args.sample_dir, f"sample{i}_watermarked.wav")
        if not (os.path.exists(clean_path) and os.path.exists(wm_path)):
            continue
        clean_wav, sr = sf.read(clean_path)
        wm_wav, wm_sr = sf.read(wm_path)
        pesq_score, stoi_score = compute_pesq_stoi(clean_wav, wm_wav, sr)
        snr_score = compute_snr(clean_wav, wm_wav)
        si_snr_score = compute_si_snr(clean_wav, wm_wav)
        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)
        snr_scores.append(snr_score)
        si_snr_scores.append(si_snr_score)
        print(f"  sample{i}: PESQ={pesq_score:.3f} (range -0.5 to 4.5, higher=better) "
              f"STOI={stoi_score:.3f} (range 0-1, higher=better) "
              f"SNR={snr_score:.2f} dB SI-SNR={si_snr_score:.2f} dB "
              f"(VoiceMark's own paper reports SI-SNR=2.01 dB -- this is the metric to compare against)")

    if pesq_scores:
        mean_pesq = sum(pesq_scores) / len(pesq_scores)
        mean_stoi = sum(stoi_scores) / len(stoi_scores)
        mean_snr = sum(snr_scores) / len(snr_scores)
        mean_si_snr = sum(si_snr_scores) / len(si_snr_scores)
        print(f"\nMean PESQ (clean vs watermarked): {mean_pesq:.3f} (VoiceMark paper: 2.20)")
        print(f"Mean STOI (clean vs watermarked): {mean_stoi:.3f} (VoiceMark paper: 0.89)")
        print(f"Mean SNR (perturbation magnitude): {mean_snr:.2f} dB")
        print(f"Mean SI-SNR (matches VoiceMark paper's exact metric): {mean_si_snr:.2f} dB (VoiceMark paper: 2.01 dB)")
        results_out["mean_pesq"] = mean_pesq
        results_out["mean_stoi"] = mean_stoi
        results_out["mean_snr_db"] = mean_snr
        results_out["mean_si_snr_db"] = mean_si_snr
        results_out["pesq_values"] = pesq_scores
        results_out["stoi_values"] = stoi_scores
        results_out["snr_values_db"] = snr_scores
        results_out["si_snr_values_db"] = si_snr_scores
    else:
        print("\nNo clean/watermarked pairs found -- check --sample_dir and --n_samples.")

    # --- WER on cloned audio ---
    if not args.skip_wer:
        print(f"\nLoading Whisper ({args.whisper_model_size})...")
        import whisper
        whisper_model = whisper.load_model(args.whisper_model_size)

        wer_scores = []
        transcriptions = []
        for i in range(args.n_samples):
            cloned_path = os.path.join(args.sample_dir, f"sample{i}_cloned.wav")
            if not os.path.exists(cloned_path):
                continue
            wer_score, hypothesis = compute_wer(cloned_path, args.reference_text, whisper_model)
            wer_scores.append(wer_score)
            transcriptions.append(hypothesis)
            print(f"  sample{i}: WER={wer_score:.3f} (0=perfect, 1.0=completely wrong) "
                  f"transcribed=\"{hypothesis}\"")

        if wer_scores:
            mean_wer = sum(wer_scores) / len(wer_scores)
            print(f"\nMean WER (cloned audio intelligibility): {mean_wer:.3f}")
            print(f"Reference text was: \"{args.reference_text}\"")
            results_out["mean_wer"] = mean_wer
            results_out["wer_values"] = wer_scores
            results_out["transcriptions"] = transcriptions
            results_out["reference_text"] = args.reference_text
        else:
            print("\nNo cloned samples found -- check --sample_dir and --n_samples.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"label": args.sample_dir, "results": results_out}, f, indent=2)
        print(f"\n[main] Saved results to {args.output}")

if __name__ == "__main__":
    main()
