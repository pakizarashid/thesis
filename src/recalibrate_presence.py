"""
src/recalibrate_presence.py

Fixes the false-positive-rate problem discovered by false_positive_rate.py:
baseline (pretrained) VoiceMark has a healthy 4% false positive rate on clean
audio, but EVERY checkpoint fine-tuned in this project (Stage 1: 76%, Stage 2:
84%) incorrectly flags most clean, never-watermarked audio as watermarked.

ROOT CAUSE: every loss used anywhere in this project (VoiceMark's own 5
losses, and SafeSpeech's disruption losses) is computed ONLY on watermarked
audio. Nothing ever penalizes the detector's presence_head (a dedicated
binary classifier, separate from the message-decoding heads) for wrongly
firing on clean audio -- there is no negative example anywhere in the
training loop, in any stage. The pretrained checkpoint's healthy calibration
apparently came entirely from VoiceMark's own original training (not
reproduced here, since their training code was never released), and
LoRA fine-tuning let it drift with nothing holding it in place.

FIX: a short additional fine-tuning pass, starting from an existing
checkpoint (Stage 1 or Stage 2), adding a binary cross-entropy loss computed
on BOTH the watermarked audio (target: presence=1) AND the SAME clean audio
run through the detector directly, bypassing embedding (target: presence=0).
This is intentionally SHORT (a few epochs, not a full 30-epoch retrain) --
the goal is recalibrating a drifted classifier head, not re-learning the
whole system from scratch.

Usage:
    python src/recalibrate_presence.py \
      --input_checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt \
      --output_dir ./checkpoints/stage1_full_recalibrated \
      --epochs 5
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from backbone import VoiceMarkBackbone
from adapters import apply_lora_adapters
from voicemark_losses import VoiceMarkLosses
from librispeech import LibriSpeechSubset, collate_librispeech


def presence_bce_loss(presence_logits: torch.Tensor, target_present: bool) -> torch.Tensor:
    """
    presence_logits: [batch, time_steps], raw logits from WMDetector's
    watermark_head (before any thresholding). target_present=True means
    every frame should predict "watermark present" (label=1); False means
    every frame should predict "no watermark" (label=0).
    """
    target = torch.ones_like(presence_logits) if target_present else torch.zeros_like(presence_logits)
    return F.binary_cross_entropy_with_logits(presence_logits, target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_checkpoint", type=str, required=True,
                    help="Existing Stage 1 or Stage 2 checkpoint to recalibrate.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=5,
                    help="Short on purpose -- recalibrating a drifted classifier head, not retraining from scratch.")
    p.add_argument("--lr", type=float, default=2e-5,
                    help="Lower than the original training lr -- avoid disturbing message-decoding "
                         "accuracy, which is already good, while fixing presence calibration.")
    p.add_argument("--lambda_presence", type=float, default=1.0,
                    help="Weight on the new presence BCE loss relative to VoiceMark's existing losses.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[recalibrate] Using device: {device}")

    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
    ckpt = torch.load(args.input_checkpoint, map_location="cpu", weights_only=False)
    backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    print(f"[recalibrate] Loaded {args.input_checkpoint} (epoch {ckpt.get('epoch')})")

    voicemark_losses_fn = VoiceMarkLosses(sample_rate=16000).to(device)

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
        epoch_fpr_flags = []
        epoch_acc = []

        for batch in train_loader:
            clean_audio = batch["waveform"].to(device)
            message = torch.randint(0, 2, (clean_audio.shape[0], 16), device=device)

            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"]

            # Positive branch: watermarked audio, exactly as all prior training did.
            detect_feat_wm = backbone.model.st_model.forward_feature(recon_wm)
            presence_logits_wm, chunk_logits_wm = backbone.model.detector(detect_feat_wm)

            vm_loss_dict = voicemark_losses_fn(
                recon_wm=recon_wm, clean_audio=clean_audio,
                acoustic=out["acoustic"], acoustic_wm=out["acoustic_wm"],
                detection_logits=presence_logits_wm, chunk_logits=chunk_logits_wm,
                message=message, discriminator=None,
            )
            positive_presence_loss = presence_bce_loss(presence_logits_wm, target_present=True)

            # NEW negative branch: the SAME clean audio, detector run directly,
            # NEVER through msg_processor -- this is the example that was
            # missing from every loss in this project until now.
            detect_feat_clean = backbone.model.st_model.forward_feature(clean_audio)
            presence_logits_clean, _ = backbone.model.detector(detect_feat_clean)
            negative_presence_loss = presence_bce_loss(presence_logits_clean, target_present=False)

            total_loss = (
                vm_loss_dict["total"]
                + args.lambda_presence * (positive_presence_loss + negative_presence_loss)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Track both metrics we care about: message accuracy shouldn't
                # regress, and false-positive flags on clean audio should drop.
                pred_chunks = torch.argmax(chunk_logits_wm, dim=-1)
                correct = 0
                total_bits = 0
                for i in range(4):
                    true_chunk = message[:, i * 4:(i + 1) * 4]
                    pred_val = pred_chunks[:, i]
                    for bit_idx in range(4):
                        pred_bit = (pred_val >> bit_idx) & 1
                        correct += (pred_bit == true_chunk[:, bit_idx]).sum().item()
                        total_bits += clean_audio.shape[0]
                epoch_acc.append(correct / total_bits)

                clean_flagged = (presence_logits_clean.mean(dim=-1) > 0).float().mean().item()
                epoch_fpr_flags.append(clean_flagged)

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[epoch {epoch} step {global_step}] total={total_loss.item():.4f} "
                      f"vm_total={vm_loss_dict['total'].item():.4f} "
                      f"pos_presence={positive_presence_loss.item():.4f} "
                      f"neg_presence={negative_presence_loss.item():.4f} "
                      f"batch_msg_acc={epoch_acc[-1]:.4f} batch_clean_flagged_rate={epoch_fpr_flags[-1]:.4f}")

        print(f"=== Epoch {epoch} complete. Msg ACC: {sum(epoch_acc)/len(epoch_acc):.4f} "
              f"| Clean-audio false-positive rate (train batches): {sum(epoch_fpr_flags)/len(epoch_fpr_flags):.4f} ===")

    lora_state = {k: v for k, v in backbone.model.state_dict().items() if "_lora." in k}
    ckpt_path = os.path.join(args.output_dir, "recalibrated_final.pt")
    torch.save({
        "epoch": ckpt.get("epoch"), "lora_state_dict": lora_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "avg_acc": sum(epoch_acc) / len(epoch_acc),  # final epoch's msg accuracy -- for schema
        # consistency with train.py/train_stage2.py's checkpoints, so downstream scripts
        # that print/expect this key don't need defensive workarounds.
        "recalibrated_from": args.input_checkpoint, "recalibration_epochs": args.epochs,
    }, ckpt_path)
    print(f"[recalibrate] Saved recalibrated checkpoint to {ckpt_path}")
    print(f"[recalibrate] Now run false_positive_rate.py and disruption_effectiveness.py "
          f"(or your usual eval) against this checkpoint to confirm FPR dropped without "
          f"regressing message decoding accuracy.")


if __name__ == "__main__":
    main()
