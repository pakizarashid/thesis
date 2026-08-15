"""
src/reduce_perturbation.py

Targets the SNR finding (~4-5 dB, low by typical watermarking standards --
see README.md Results Section 5): the watermark is a real, non-trivial
perturbation, not a subtle one. This is a loss-WEIGHTING issue, not a
data-scarcity issue -- VoiceMark's own loss already includes perceptual-
similarity terms (Lmel, Lcos) alongside the decode-accuracy term (Ldec); the
low SNR suggests Ldec is currently winning that balance more than necessary.

Method: a SHORT additional fine-tuning pass (not a full retrain) from the
canonical, already-validated checkpoint, with lambda_mel and lambda_cos
increased relative to their defaults (paper values: lambda_vad=1, lambda_cos=2,
lambda_mel=2, lambda_adv=1, lambda_dec=1), while closely monitoring message
decode accuracy every epoch -- the goal is to find how much perceptual
improvement is available before decode accuracy starts degrading
meaningfully, not to blindly maximize imperceptibility at accuracy's expense.

Usage:
    python src/reduce_perturbation.py \
      --input_checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \
      --output_dir ./checkpoints/stage1_low_perturbation \
      --lambda_mel 4.0 --lambda_cos 4.0 --epochs 5
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from voicemark_losses import VoiceMarkLosses
from librispeech import LibriSpeechSubset, collate_librispeech


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_checkpoint", type=str, required=True,
                    help="Existing checkpoint to continue from -- use your canonical, "
                         "already-validated checkpoint, not a fresh start.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=5,
                    help="Short on purpose -- rebalancing an already-trained model, not retraining from scratch.")
    p.add_argument("--lr", type=float, default=2e-5,
                    help="Low, matching recalibrate_presence.py's approach -- gentle adjustment, not aggressive retraining.")
    # VoiceMark's own paper defaults: lambda_vad=1, lambda_cos=2, lambda_mel=2, lambda_adv=1, lambda_dec=1
    p.add_argument("--lambda_vad", type=float, default=1.0)
    p.add_argument("--lambda_cos", type=float, default=4.0, help="Increased from paper default (2.0) toward perceptual similarity.")
    p.add_argument("--lambda_mel", type=float, default=4.0, help="Increased from paper default (2.0) toward perceptual similarity.")
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_dec", type=float, default=1.0, help="Unchanged -- decode accuracy still matters, just no longer over-weighted.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=30)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[reduce_perturbation] Using device: {device}")
    print(f"[reduce_perturbation] Loss weights: vad={args.lambda_vad} cos={args.lambda_cos} "
          f"mel={args.lambda_mel} adv={args.lambda_adv} dec={args.lambda_dec} "
          f"(paper defaults: vad=1 cos=2 mel=2 adv=1 dec=1)")

    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
    ckpt = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
    backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    print(f"[reduce_perturbation] Loaded {args.input_checkpoint} (epoch {ckpt.get('epoch')})")

    voicemark_losses_fn = VoiceMarkLosses(
        sample_rate=16000, lambda_vad=args.lambda_vad, lambda_cos=args.lambda_cos,
        lambda_mel=args.lambda_mel, lambda_adv=args.lambda_adv, lambda_dec=args.lambda_dec,
    ).to(device)

    train_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="train",
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_librispeech)

    trainable_params = [p_ for p_ in backbone.model.parameters() if p_.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    backbone.model.train()

    global_step = 0
    for epoch in range(args.epochs):
        epoch_acc = []
        epoch_mel_loss = []
        epoch_waveform_l1 = []  # cheap proxy for perturbation magnitude, tracked live

        for batch in train_loader:
            clean_audio = batch["waveform"].to(device)
            message = torch.randint(0, 2, (clean_audio.shape[0], 16), device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]
            detect_feat = backbone.model.st_model.forward_feature(recon_wm)
            presence_logits, chunk_logits = backbone.model.detector(detect_feat)

            vm_loss_dict = voicemark_losses_fn(
                recon_wm=recon_wm, clean_audio=clean_audio,
                acoustic=out["acoustic"], acoustic_wm=out["acoustic_wm"],
                detection_logits=presence_logits, chunk_logits=chunk_logits,
                message=message, discriminator=None,
            )

            optimizer.zero_grad()
            vm_loss_dict["total"].backward()
            optimizer.step()

            with torch.no_grad():
                pred_chunks = torch.argmax(chunk_logits, dim=-1)
                correct, total_bits = 0, 0
                for i in range(4):
                    true_chunk = message[:, i * 4:(i + 1) * 4]
                    pred_val = pred_chunks[:, i]
                    for bit_idx in range(4):
                        pred_bit = (pred_val >> bit_idx) & 1
                        correct += (pred_bit == true_chunk[:, bit_idx]).sum().item()
                        total_bits += clean_audio.shape[0]
                epoch_acc.append(correct / total_bits)

                # Cheap live proxy for perturbation magnitude (NOT the real SNR --
                # just a fast per-step signal to watch the trend during training;
                # run quality_metrics.py for the real, trustworthy measurement
                # after training completes).
                waveform_l1 = (recon_wm - clean_audio).abs().mean().item()
                epoch_waveform_l1.append(waveform_l1)
                epoch_mel_loss.append(vm_loss_dict.get("lmel", torch.tensor(0.0)).item()
                                       if "lmel" in vm_loss_dict else 0.0)

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[epoch {epoch} step {global_step}] total={vm_loss_dict['total'].item():.4f} "
                      f"msg_acc={epoch_acc[-1]:.4f} waveform_l1_proxy={epoch_waveform_l1[-1]:.6f}")

        print(f"=== Epoch {epoch} complete. Msg ACC: {sum(epoch_acc)/len(epoch_acc):.4f} "
              f"| Mean waveform L1 (perturbation proxy -- lower = smaller perturbation): "
              f"{sum(epoch_waveform_l1)/len(epoch_waveform_l1):.6f} ===")

    lora_state = {k: v for k, v in backbone.model.state_dict().items() if "_lora." in k}
    ckpt_path = os.path.join(args.output_dir, "low_perturbation_final.pt")
    torch.save({
        "epoch": ckpt.get("epoch"), "lora_state_dict": lora_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "avg_acc": sum(epoch_acc) / len(epoch_acc),
        "rebalanced_from": args.input_checkpoint,
        "loss_weights": {"vad": args.lambda_vad, "cos": args.lambda_cos, "mel": args.lambda_mel,
                          "adv": args.lambda_adv, "dec": args.lambda_dec},
    }, ckpt_path)
    print(f"[reduce_perturbation] Saved checkpoint to {ckpt_path}")
    print(f"[reduce_perturbation] Now run save_audio_samples.py + quality_metrics.py against this "
          f"checkpoint to measure REAL SNR/PESQ/STOI, and disruption_effectiveness.py or "
          f"audiopure_eval.py's acc_before to confirm message decode accuracy didn't regress meaningfully.")


if __name__ == "__main__":
    main()
