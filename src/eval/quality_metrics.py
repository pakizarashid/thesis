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
import numpy as np
import soundfile as sf
import librosa


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
    """
    import jiwer

    result = whisper_model.transcribe(audio_path, language="en")
    hypothesis = result["text"].strip()

    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    wer_score = jiwer.wer(reference_text, hypothesis,
                           truth_transform=transform, hypothesis_transform=transform)
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
    args = p.parse_args()

    print(f"\n{'=' * 60}\nQuality metrics for: {args.sample_dir}\n{'=' * 60}")

    # --- PESQ / STOI on clean vs watermarked ---
    pesq_scores, stoi_scores = [], []
    for i in range(args.n_samples):
        clean_path = os.path.join(args.sample_dir, f"sample{i}_clean.wav")
        wm_path = os.path.join(args.sample_dir, f"sample{i}_watermarked.wav")
        if not (os.path.exists(clean_path) and os.path.exists(wm_path)):
            continue
        clean_wav, sr = sf.read(clean_path)
        wm_wav, wm_sr = sf.read(wm_path)
        pesq_score, stoi_score = compute_pesq_stoi(clean_wav, wm_wav, sr)
        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)
        print(f"  sample{i}: PESQ={pesq_score:.3f} (range -0.5 to 4.5, higher=better) "
              f"STOI={stoi_score:.3f} (range 0-1, higher=better)")

    if pesq_scores:
        print(f"\nMean PESQ (clean vs watermarked): {sum(pesq_scores)/len(pesq_scores):.3f}")
        print(f"Mean STOI (clean vs watermarked): {sum(stoi_scores)/len(stoi_scores):.3f}")
    else:
        print("\nNo clean/watermarked pairs found -- check --sample_dir and --n_samples.")

    # --- WER on cloned audio ---
    if not args.skip_wer:
        print(f"\nLoading Whisper ({args.whisper_model_size})...")
        import whisper
        whisper_model = whisper.load_model(args.whisper_model_size)

        wer_scores = []
        for i in range(args.n_samples):
            cloned_path = os.path.join(args.sample_dir, f"sample{i}_cloned.wav")
            if not os.path.exists(cloned_path):
                continue
            wer_score, hypothesis = compute_wer(cloned_path, args.reference_text, whisper_model)
            wer_scores.append(wer_score)
            print(f"  sample{i}: WER={wer_score:.3f} (0=perfect, 1.0=completely wrong) "
                  f"transcribed=\"{hypothesis}\"")

        if wer_scores:
            print(f"\nMean WER (cloned audio intelligibility): {sum(wer_scores)/len(wer_scores):.3f}")
            print(f"Reference text was: \"{args.reference_text}\"")
        else:
            print("\nNo cloned samples found -- check --sample_dir and --n_samples.")


if __name__ == "__main__":
    main()
