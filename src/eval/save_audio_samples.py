"""
src/eval/save_audio_samples.py

Saves real, playable .wav files (clean original, watermarked, and the
surrogate's cloned output) so you can LISTEN directly and check whether the
pivotal_distance/SIM disagreement (Stage 2 longrun: pivotal jumped +0.20 but
SIM barely moved) reflects genuine audio quality collapse (garbled/noisy
output that increases mel-distance without specifically disrupting speaker
identity) versus something else entirely. No automated metric substitutes for
actually listening here -- this is the ground-truth check.

One model per process (same safe pattern as every other eval script in this
project -- loading two backbones sequentially in one process previously
corrupted GPU state).

Usage:
    python src/eval/save_audio_samples.py --output_dir ./audio_samples/baseline
    python src/eval/save_audio_samples.py --checkpoint ./checkpoints/stage2_longrun/stage2_epoch29.pt --output_dir ./audio_samples/stage2_longrun
"""

import os
import sys
import argparse
import torch
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from librispeech import LibriSpeechSubset, collate_librispeech
from torch.utils.data import DataLoader


def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16):
    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=r, alpha=alpha)
    if lora_checkpoint_path is not None:
        ckpt = torch.load(lora_checkpoint_path, map_location="cpu", weights_only=False)
        backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[build_backbone] Loaded checkpoint {lora_checkpoint_path} (epoch {ckpt.get('epoch')})")
    else:
        print("[build_backbone] Using baseline (LoRA zero-init)")
    return backbone


def save_wav(path: str, waveform: torch.Tensor, sample_rate: int = 16000):
    """waveform: [1, T] or [T] tensor -> writes a playable wav file."""
    wav_np = waveform.detach().cpu().squeeze().numpy()
    sf.write(path, wav_np, sample_rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--n_samples", type=int, default=4, help="How many eval utterances to save.")
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)
    surrogate = load_yourtts_surrogate(device=device)
    backbone.model.eval()

    print(f"[main] Saving {args.n_samples} samples to {args.output_dir}/ ...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_samples:
                break
            clean_audio = batch["waveform"].to(device)
            message = torch.randint(0, 2, (1, 16), device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            cloned_output = surrogate.clone_voice(recon_wm, text=args.surrogate_text)

            save_wav(os.path.join(args.output_dir, f"sample{i}_clean.wav"), clean_audio[0])
            save_wav(os.path.join(args.output_dir, f"sample{i}_watermarked.wav"), recon_wm[0])
            save_wav(os.path.join(args.output_dir, f"sample{i}_cloned.wav"), cloned_output[0])

            print(f"  sample{i}: speaker={batch['speaker_id'][0]}, "
                  f"clean_len={clean_audio.shape[-1]}, cloned_len={cloned_output.shape[-1]}")

    print(f"[main] Done. Files saved in {args.output_dir}/")
    print("Listen to sampleN_clean.wav vs sampleN_cloned.wav for each N -- "
          "does the cloned version sound like garbled noise (quality collapse) "
          "or like coherent speech, possibly in a different voice (targeted disruption)?")


if __name__ == "__main__":
    main()
