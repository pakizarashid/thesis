"""
src/eval/vad_watermark_probability_viz.py

Reproduces VoiceMark's own Figure 3 style: a mel spectrogram with a color
band above it showing the detector's per-frame watermark-detection
PROBABILITY (red = high, blue = low), aligned to VAD-segmented speech
regions.

THIS IS NOT audio_diff_analysis.py. That script answers "is the watermark
AUDIBLE" (clean-vs-watermarked difference spectrogram). This script answers
a different question: "WHERE in time is the watermark actually being
detected, and does that track real speech vs. silence the way it should" --
the same question VoiceMark's Figure 3 visualizes, and a natural sanity
check on the false_positive_rate.py finding (this project's own fine-tuned
checkpoints drifted toward firing on clean audio -- this figure lets you
SEE whether a given checkpoint's presence probability tracks VAD-detected
speech tightly, or fires indiscriminately including over silence).

Per-frame probability requires no new inference machinery: WMDetector's
presence head already outputs one logit per codec-latent frame (the exact
tensor false_positive_rate.py already thresholds, and Lvad already trains
against -- see src/losses/voicemark_losses.py::compute_lvad). So
sigmoid(presence_logits) IS the per-frame probability array; no
sliding-window pass is needed.

VAD reuses rabiner_dual_threshold_vad() from voicemark_losses.py -- the same
implementation used for Lvad during training. Per that module's own
documented confidence level: this is a reconstruction of the classical
Rabiner (1978) dual-threshold algorithm the paper cites by name, NOT a
verified reproduction of VoiceMark's own (unavailable) VAD code. State this
caveat in the thesis alongside this figure, exactly as already done
elsewhere for Lvad.

One model per process (same safe pattern as every other eval script here).

Usage:
    # Fresh eval utterances, embeds + detects itself:
    python src/eval/vad_watermark_probability_viz.py \\
        --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --output_dir ./figures/vad_probability_band --n_samples 4

    # Baseline (pretrained, no fine-tuning) for comparison:
    python src/eval/vad_watermark_probability_viz.py \\
        --output_dir ./figures/vad_probability_band_baseline --n_samples 4

    # Against an existing save_audio_samples.py output directory (re-decodes
    # the already-saved watermarked wav rather than re-embedding):
    python src/eval/vad_watermark_probability_viz.py \\
        --checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --sample_dir ./audio_samples/proof_stage1 --output_dir ./figures/vad_probability_band
"""

import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from librispeech import LibriSpeechSubset, collate_librispeech
from torch.utils.data import DataLoader
from voicemark_losses import rabiner_dual_threshold_vad, align_vad_to_latent_frames


def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16):
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)
    if lora_checkpoint_path is not None:
        ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[build_backbone] Loaded checkpoint {lora_checkpoint_path}")
    else:
        print("[build_backbone] Using baseline (LoRA zero-init, == pretrained VoiceMark)")
    return backbone


def get_presence_probability(backbone, audio: torch.Tensor) -> np.ndarray:
    """audio: [1, 1, T] waveform on the model's device. Returns [n_frames] probs."""
    with torch.no_grad():
        detect_feat = backbone.model.st_model.forward_feature(audio)
        presence_logits, _ = backbone.model.detector(detect_feat)  # [1, n_frames]
    return torch.sigmoid(presence_logits)[0].detach().cpu().numpy()


def get_vad_mask(clean_audio: torch.Tensor, sample_rate: int, target_n_frames: int) -> np.ndarray:
    """clean_audio: [1, 1, T] or [1, T]. Returns [target_n_frames] binary array,
    aligned to the SAME frame count as the detector's presence probability so
    the two can be plotted on a shared axis."""
    vad_raw = rabiner_dual_threshold_vad(clean_audio, sample_rate)  # [1, n_vad_frames]
    vad_aligned = align_vad_to_latent_frames(vad_raw, target_n_frames)  # [1, target_n_frames]
    return vad_aligned[0].detach().cpu().numpy()


def make_figure(watermarked_wav: np.ndarray, sr: int, probs: np.ndarray, vad_mask: np.ndarray,
                 out_path: str, title_suffix: str = ""):
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=watermarked_wav.astype(np.float32), sr=sr,
                                        n_fft=1024, hop_length=256, n_mels=80),
        ref=np.max,
    )
    n_mel_frames = mel.shape[1]

    # probs and vad_mask live at the codec's latent frame rate, mel lives at
    # the STFT hop rate -- these are two independent framings with different
    # hop sizes, so interpolate both onto the mel spectrogram's own frame
    # axis (by fractional position, not frame count) purely for DISPLAY
    # alignment. This does not change either underlying measurement.
    x_probs = np.linspace(0, 1, len(probs))
    x_vad = np.linspace(0, 1, len(vad_mask))
    x_mel = np.linspace(0, 1, n_mel_frames)
    probs_interp = np.interp(x_mel, x_probs, probs)
    vad_interp = np.interp(x_mel, x_vad, vad_mask)

    fig, (ax_band, ax_mel) = plt.subplots(
        2, 1, figsize=(12, 5), gridspec_kw={"height_ratios": [1, 6]}, sharex=True
    )

    # Watermark-probability color band: red = high probability, blue = low --
    # matches VoiceMark's own Figure 3 convention.
    cmap = LinearSegmentedColormap.from_list("wm_prob", ["#2166ac", "#f7f7f7", "#b2182b"])
    band = probs_interp.reshape(1, -1)
    im_band = ax_band.imshow(band, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                              extent=[0, n_mel_frames, 0, 1])
    ax_band.set_yticks([])
    ax_band.set_ylabel("P(wm)", rotation=0, labelpad=25, va="center")
    ax_band.set_title(f"Watermark detection probability (top band, red=high/blue=low) "
                       f"+ VAD-aligned mel spectrogram{title_suffix}")

    # Mark VAD speech/silence transitions as vertical lines on both panels --
    # this is the actual diagnostic value of the figure: do high-probability
    # regions track real speech (expected), or fire during silence too (a
    # real failure mode -- see false_positive_rate.py's drifted-checkpoint
    # finding)?
    vad_binary = (vad_interp > 0.5).astype(int)
    transitions = np.where(np.diff(vad_binary) != 0)[0]
    for t in transitions:
        ax_band.axvline(t, color="black", linewidth=0.6, alpha=0.5)
        ax_mel.axvline(t, color="cyan", linewidth=0.6, alpha=0.5, linestyle="--")

    im_mel = ax_mel.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                            extent=[0, n_mel_frames, 0, mel.shape[0]])
    ax_mel.set_xlabel("time frames (mel-hop resolution; cyan dashed = VAD speech/silence boundary)")
    ax_mel.set_ylabel("mel bin")

    cbar = plt.colorbar(im_band, ax=ax_band, fraction=0.046, pad=0.02)
    cbar.set_label("P(watermark present)")
    plt.colorbar(im_mel, ax=ax_mel, fraction=0.046, pad=0.02, label="dB")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def run_from_dataset(backbone, device, args):
    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    label = "baseline" if args.checkpoint is None else os.path.basename(os.path.dirname(args.checkpoint))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_samples:
                break
            clean_audio = batch["waveform"].to(device)
            message = torch.randint(0, 2, (1, 16), device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]

            probs = get_presence_probability(backbone, recon_wm)
            vad_mask = get_vad_mask(clean_audio, sample_rate=16000, target_n_frames=len(probs))

            watermarked_wav = recon_wm[0].detach().cpu().squeeze().numpy()
            out_path = os.path.join(args.output_dir, f"{label}_sample{i}_vad_prob_band.png")
            make_figure(watermarked_wav, sr=16000, probs=probs, vad_mask=vad_mask,
                        out_path=out_path, title_suffix=f" [{label}, sample {i}]")


def run_from_sample_dir(backbone, device, args):
    for i in args.sample_indices:
        watermarked_path = os.path.join(args.sample_dir, f"sample{i}_watermarked.wav")
        clean_path = os.path.join(args.sample_dir, f"sample{i}_clean.wav")
        if not os.path.exists(watermarked_path):
            print(f"  [skip] {watermarked_path} not found")
            continue

        watermarked_wav, sr = sf.read(watermarked_path)
        clean_wav, clean_sr = sf.read(clean_path)
        assert sr == clean_sr, "clean/watermarked sample rates must match"

        wm_tensor = torch.from_numpy(watermarked_wav).float().unsqueeze(0).unsqueeze(0).to(device)
        clean_tensor = torch.from_numpy(clean_wav).float().unsqueeze(0).unsqueeze(0).to(device)

        probs = get_presence_probability(backbone, wm_tensor)
        vad_mask = get_vad_mask(clean_tensor, sample_rate=sr, target_n_frames=len(probs))

        label = os.path.basename(os.path.normpath(args.sample_dir))
        out_path = os.path.join(args.output_dir, f"{label}_sample{i}_vad_prob_band.png")
        make_figure(watermarked_wav, sr=sr, probs=probs, vad_mask=vad_mask,
                    out_path=out_path, title_suffix=f" [{label}, sample {i}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    # Mode A: fresh utterances straight from the eval split
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--n_samples", type=int, default=4)

    # Mode B: re-use save_audio_samples.py's output directory instead
    p.add_argument("--sample_dir", type=str, default=None,
                    help="If set, reads sampleN_clean.wav / sampleN_watermarked.wav from here "
                         "(save_audio_samples.py's output format) instead of the dataset.")
    p.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="Which sample indices to use when --sample_dir is set.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    backbone.model.eval()

    if args.sample_dir is not None:
        run_from_sample_dir(backbone, device, args)
    else:
        run_from_dataset(backbone, device, args)

    print(f"[main] Done. Figures saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
