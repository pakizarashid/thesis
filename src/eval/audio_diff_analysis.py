"""
src/eval/audio_diff_analysis.py

Produces an accurate, quantitative comparison of clean/watermarked/cloned
audio -- not just spectrograms to eyeball.

IMPORTANT DISTINCTION (read before interpreting the plots):
  - clean vs watermarked: SAME content, same duration, frame-aligned. A
    direct difference spectrogram is meaningful here -- it should look like
    near-zero/uniform noise if the watermark is perceptually transparent.
  - clean vs cloned: DIFFERENT content (the surrogate synthesizes a fixed
    placeholder sentence, not the original words), different duration. A
    frame-by-frame difference plot would be MEANINGLESS here -- there is no
    valid alignment between "hello world" and "this is a test sentence".
    Instead, this script reports: (a) side-by-side spectrograms for visual
    comparison, (b) SIM (speaker embedding cosine similarity -- the accurate,
    content-independent answer to "how different does this voice sound"),
    (c) objective descriptive stats (duration, RMS energy, pitch range) that
    describe HOW the audio differs without requiring alignment.

Usage:
    python src/eval/audio_diff_analysis.py --sample_dir ./audio_samples/baseline --sample_idx 0
    python src/eval/audio_diff_analysis.py --sample_dir ./audio_samples/stage2_longrun --sample_idx 0
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def load_mel(wav, sr, n_fft=1024, hop=256, n_mels=80):
    mel = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=sr, n_fft=n_fft,
                                          hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def descriptive_stats(wav, sr):
    duration = len(wav) / sr
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    try:
        f0, voiced_flag, _ = librosa.pyin(wav.astype(np.float32), sr=sr,
                                           fmin=librosa.note_to_hz("C2"),
                                           fmax=librosa.note_to_hz("C7"))
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else float("nan")
        pitch_std = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 0 else float("nan")
    except Exception as e:
        pitch_mean, pitch_std = float("nan"), float("nan")
        print(f"  (pitch estimation failed: {e})")
    return {"duration_s": duration, "rms_energy": rms, "pitch_mean_hz": pitch_mean, "pitch_std_hz": pitch_std}


def analyze_pair_same_content(clean_wav, other_wav, sr, other_label: str, out_prefix: str):
    """clean vs watermarked -- same content, alignable, direct difference is meaningful."""
    min_len = min(len(clean_wav), len(other_wav))
    clean_wav = clean_wav[:min_len]
    other_wav = other_wav[:min_len]

    clean_mel = load_mel(clean_wav, sr)
    other_mel = load_mel(other_wav, sr)
    min_frames = min(clean_mel.shape[1], other_mel.shape[1])
    clean_mel = clean_mel[:, :min_frames]
    other_mel = other_mel[:, :min_frames]
    diff_mel = other_mel - clean_mel

    waveform_l2 = float(np.sqrt(np.mean((clean_wav - other_wav) ** 2)))
    mel_l1 = float(np.mean(np.abs(diff_mel)))
    correlation = float(np.corrcoef(clean_wav, other_wav)[0, 1])

    print(f"\n--- clean vs {other_label} (same content -- direct comparison valid) ---")
    print(f"  Waveform RMS difference: {waveform_l2:.6f}")
    print(f"  Mel-spectrogram mean abs difference (dB): {mel_l1:.4f}")
    print(f"  Waveform correlation coefficient: {correlation:.6f} (1.0 = identical)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    panels = [
        (clean_mel, "clean", "magma", None, None),
        (other_mel, other_label, "magma", None, None),
        (diff_mel, f"difference ({other_label} - clean)", "coolwarm", -20, 20),
    ]
    for ax, (mel, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time frames")
        ax.set_ylabel("mel bin")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"clean vs {other_label} -- mel-spectrogram and difference (dB)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_clean_vs_{other_label}_diff.png", dpi=150)
    plt.show()
    print(f"  Saved {out_prefix}_clean_vs_{other_label}_diff.png")


def analyze_pair_different_content(clean_wav, cloned_wav, sr, out_prefix: str,
                                    sim_value: float = None, pivotal_value: float = None):
    """clean vs cloned -- different content, NOT alignable, no direct diff plot."""
    clean_mel = load_mel(clean_wav, sr)
    cloned_mel = load_mel(cloned_wav, sr)

    clean_stats = descriptive_stats(clean_wav, sr)
    cloned_stats = descriptive_stats(cloned_wav, sr)

    print(f"\n--- clean vs cloned (DIFFERENT content -- no frame-alignment possible, "
          f"direct diff plot would be meaningless) ---")
    print(f"  {'':20s} {'clean':>12s} {'cloned':>12s}")
    for key in ["duration_s", "rms_energy", "pitch_mean_hz", "pitch_std_hz"]:
        print(f"  {key:20s} {clean_stats[key]:>12.4f} {cloned_stats[key]:>12.4f}")
    if sim_value is not None:
        print(f"\n  SIM (speaker similarity, content-independent, THE accurate answer "
              f"to 'how different does the voice sound'): {sim_value:.4f}")
        print(f"  (1.0 = identical speaker identity as perceived by the encoder, "
              f"0.0 = completely different)")
    if pivotal_value is not None:
        print(f"  Pivotal mel-distance (raw spectral distance, NOT speaker-identity-specific): "
              f"{pivotal_value:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mel, title in zip(axes, [clean_mel, cloned_mel], ["clean (original speaker)", "cloned (surrogate output)"]):
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("time frames")
        ax.set_ylabel("mel bin")
        plt.colorbar(im, ax=ax, fraction=0.046)
    subtitle = ""
    if sim_value is not None:
        subtitle = f"  |  SIM={sim_value:.4f}"
    plt.suptitle(f"clean vs cloned -- different content, side-by-side only{subtitle}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_clean_vs_cloned_sidebyside.png", dpi=150)
    plt.show()
    print(f"  Saved {out_prefix}_clean_vs_cloned_sidebyside.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", type=str, required=True,
                    help="Directory containing sampleN_clean.wav, sampleN_watermarked.wav, sampleN_cloned.wav "
                         "(from save_audio_samples.py)")
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--sim_value", type=float, default=None,
                    help="Optional: paste the SIM value from disruption_effectiveness.py's output for this "
                         "condition, to display it alongside the plots.")
    p.add_argument("--pivotal_value", type=float, default=None,
                    help="Optional: paste the pivotal_distance value from disruption_effectiveness.py's output.")
    args = p.parse_args()

    clean_path = os.path.join(args.sample_dir, f"sample{args.sample_idx}_clean.wav")
    watermarked_path = os.path.join(args.sample_dir, f"sample{args.sample_idx}_watermarked.wav")
    cloned_path = os.path.join(args.sample_dir, f"sample{args.sample_idx}_cloned.wav")

    clean_wav, sr = sf.read(clean_path)
    watermarked_wav, wm_sr = sf.read(watermarked_path)
    cloned_wav, cloned_sr = sf.read(cloned_path)

    out_prefix = os.path.basename(os.path.normpath(args.sample_dir)) + f"_sample{args.sample_idx}"

    analyze_pair_same_content(clean_wav, watermarked_wav, sr, "watermarked", out_prefix)
    analyze_pair_different_content(clean_wav, cloned_wav, sr, out_prefix,
                                    sim_value=args.sim_value, pivotal_value=args.pivotal_value)


if __name__ == "__main__":
    main()
