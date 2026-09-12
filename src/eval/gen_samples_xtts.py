"""
src/eval/gen_samples_xtts.py

Same purpose as gen_samples_yourtts.py, for XTTS-v2. Writes sample{i}_reference.wav
(= recon_wm) and sample{i}_clone_xtts.wav (= XTTS's clone of it), same naming
convention cloner_watermark_eval.py already uses. Reuses load_xtts/xtts_clone
directly from xtts_transfer_eval.py (already verified there) -- this script adds
only the save-to-disk step that file never needed.

TEXT: same reasoning as gen_samples_yourtts.py -- synthesizes the reference's own
LibriSpeech transcript by default (not the generic fixed sentence every other
script in this project uses) so CARRIER-PROBE's frame-by-frame latent comparison
isn't confounded by source and clone saying different words.

CROP_SECONDS (2026-09-12 fix): also bumped from the project's usual 3.0s default to
20.0s, for the same reason as gen_samples_yourtts.py -- LibriSpeechSubset's
transcript covers the FULL utterance but its waveform is cropped/center-cropped to
a fixed length, so at 3.0s the reference audio was often only a middle slice of
what the transcript describes. See that file's docstring for the full explanation.

TRAILING-SILENCE TRIM (2026-09-12, second fix): also ported from
gen_samples_yourtts.py -- the 20.0s crop pads most (shorter) LibriSpeech
utterances with several seconds of trailing silence, which carrier_probe.py's
frame-count-mismatch check was misreading as a content mismatch. Trimmed at the
source via trim_trailing_silence() instead of trying to compensate downstream.

Usage:
    python src/eval/gen_samples_xtts.py --n_utterances 15 \\
        --save_clones_dir ./audio_samples/carrierprobe_xtts
"""
import os
import sys
import argparse
import soundfile as sf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from disruption_pgd import build_backbone, random_message, detect_acc
from xtts_transfer_eval import load_xtts, xtts_clone
from librispeech import LibriSpeechSubset, collate_librispeech


def trim_trailing_silence(waveform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Same as gen_samples_yourtts.py's -- see that file's docstring for why this exists."""
    w = waveform.squeeze(0) if waveform.dim() > 1 else waveform
    nonzero = (w.abs() > eps).nonzero()
    if nonzero.numel() == 0:
        return waveform
    last = nonzero[-1].item()
    trimmed = w[: last + 1]
    return trimmed.unsqueeze(0) if waveform.dim() > 1 else trimmed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Omit for pretrained VoiceMark -- matches the ladder's own protocol.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--n_utterances", type=int, default=15)
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.",
                    help="Fallback ONLY if a reference utterance's own transcript is empty.")
    p.add_argument("--use_own_transcript", action="store_true", default=True,
                    help="Synthesize the reference's own LibriSpeech transcript instead of "
                         "--surrogate_text (default True). Pass --no_use_own_transcript to "
                         "restore the old fixed-sentence behavior.")
    p.add_argument("--no_use_own_transcript", dest="use_own_transcript", action="store_false")
    p.add_argument("--save_clones_dir", type=str, required=True)
    p.add_argument("--tmp_dir", type=str, default="./tmp_xtts_refs_carrierprobe")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=20.0,
                    help="Bumped up from the project's usual 3.0s default -- see the "
                         "CROP_SECONDS note in gen_samples_yourtts.py's docstring.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, False, 32)
    tts = load_xtts(device)

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    os.makedirs(args.save_clones_dir, exist_ok=True)
    n_written = 0
    for i, batch in enumerate(loader):
        if i >= args.n_utterances:
            break
        clean_audio = trim_trailing_silence(batch["waveform"][0]).unsqueeze(0).to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=args.message_seed_base + i)

        text = args.surrogate_text
        if args.use_own_transcript:
            own_transcript = (batch.get("transcript") or [""])[0].strip()
            if own_transcript:
                text = own_transcript
            elif i == 0:
                print(f"[gen_samples_xtts] WARNING: empty transcript for utterance 0 -- "
                      f"falling back to --surrogate_text ({args.surrogate_text!r}).")

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()
        cloned = xtts_clone(tts, recon_wm, text, args.tmp_dir, tag=f"cp{i}")

        if i == 0:
            a_src = detect_acc(backbone, recon_wm, message)
            print(f"[gen_samples_xtts] sample 0 acc_source={a_src:.4f} (expect ~0.99).")
            if a_src < 0.85:
                raise SystemExit("ABORT: acc_source too low -- fix checkpoint/seed before saving.")

        sf.write(os.path.join(args.save_clones_dir, f"sample{i}_reference.wav"),
                 recon_wm[0].detach().cpu().reshape(-1).numpy(), 16000)
        sf.write(os.path.join(args.save_clones_dir, f"sample{i}_clone_xtts.wav"),
                 cloned[0].detach().cpu().reshape(-1).numpy(), 16000)
        n_written += 1

    print(f"[gen_samples_xtts] wrote {n_written} reference/clone pairs to {args.save_clones_dir}")


if __name__ == "__main__":
    main()
