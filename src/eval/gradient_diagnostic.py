"""
src/eval/gradient_diagnostic.py

Diagnoses WHY scaling lambda_disrupt_max (0.01 -> 0.1, a 10x jump) produced no
measurable change in SIM after full training runs. The hypothesis: loss VALUE
magnitude and GRADIENT magnitude at the LoRA parameters are not the same
thing. The disruption loss must backpropagate through the surrogate's full
frozen ResNet speaker encoder, then the VITS flow/decoder, then the frozen
SpeechTokenizer decoder, before reaching the 294K trainable LoRA parameters --
a much longer chain than the VoiceMark losses' 1-2 hop path to the same
parameters. A large loss VALUE doesn't guarantee a large GRADIENT if it's
attenuated along that chain.

This script computes ONE forward pass, then backpropagates vm_loss and
disrupt_loss SEPARATELY (each with retain_graph=True where needed), measuring
the L2 norm of the resulting gradients on the LoRA parameters for each. The
RATIO of these norms tells us how much lambda would ACTUALLY need to scale to
make the disruption loss's gradient contribution comparable to the VoiceMark
losses' -- which may be very different from what loss-VALUE matching alone
would suggest (and is the real quantity that determines whether training
changes the weights meaningfully).

Usage:
    python src/eval/gradient_diagnostic.py --stage1_checkpoint ./checkpoints/stage1_full/stage1_epoch29.pt
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from backbone import VoiceMarkBackbone, VoiceMarkDiscriminator
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from voicemark_losses import VoiceMarkLosses
from safespeech_losses import SafeSpeechDisruptionLoss
from librispeech import LibriSpeechSubset, collate_librispeech
from torch.utils.data import DataLoader


def lora_param_grad_norm(backbone) -> float:
    """L2 norm of gradients across all LoRA parameters (pools every LoRA
    tensor's gradient into one combined norm, standard total-norm convention)."""
    total_sq = 0.0
    for name, p in backbone.model.named_parameters():
        if "_lora." in name and p.grad is not None:
            total_sq += p.grad.detach().norm(2).item() ** 2
    return total_sq ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--surrogate_sample_rate", type=int, default=16000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diagnostic] Using device: {device}")

    backbone = VoiceMarkBackbone()
    apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
    ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    print(f"[diagnostic] Loaded Stage 1 checkpoint (epoch {ckpt.get('epoch')})")

    discriminator = VoiceMarkDiscriminator()
    surrogate = load_yourtts_surrogate(device=device)

    train_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="train",
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_librispeech, drop_last=True)
    batch = next(iter(loader))
    clean_audio = batch["waveform"].to(device)
    message = torch.randint(0, 2, (clean_audio.shape[0], 16), device=device)

    voicemark_losses_fn = VoiceMarkLosses(sample_rate=16000).to(device)
    disruption_loss_fn = SafeSpeechDisruptionLoss(sampling_rate=args.surrogate_sample_rate).to(device)

    backbone.model.train()

    # --- Pass 1: VoiceMark losses only ---
    trainable_params = [p_ for p_ in backbone.model.parameters() if p_.requires_grad]
    for p_ in trainable_params:
        if p_.grad is not None:
            p_.grad = None

    out = backbone.forward_full(clean_audio, message)
    recon_wm = out["recon_wm"]
    detect_feat = backbone.model.st_model.forward_feature(recon_wm)
    logits, chunk_logits = backbone.model.detector(detect_feat)

    vm_loss_dict = voicemark_losses_fn(
        recon_wm=recon_wm, clean_audio=clean_audio,
        acoustic=out["acoustic"], acoustic_wm=out["acoustic_wm"],
        detection_logits=logits, chunk_logits=chunk_logits,
        message=message, discriminator=discriminator,
    )
    vm_loss_dict["total"].backward()
    vm_grad_norm = lora_param_grad_norm(backbone)
    print(f"\n[diagnostic] VoiceMark losses total = {vm_loss_dict['total'].item():.4f}")
    print(f"[diagnostic] VoiceMark losses -> LoRA gradient norm = {vm_grad_norm:.8f}")

    # --- Pass 2: disruption loss, broken down by SUB-COMPONENT ---
    # Tests whether the large combined gradient is dominated by kl_to_noise/
    # l1_to_noise (which push mel statistics toward literal random noise, a
    # "loud" but possibly SIM-irrelevant objective) rather than
    # pivotal_disruption (the term that actually drives speaker similarity).
    #
    # IMPORTANT: each sub-loss is multiplied by its ACTUAL CONFIGURED WEIGHT
    # (disruption_loss_fn.lambda_mel / .weight_kl / .weight_l1) before
    # measuring gradient contribution -- an earlier version of this script
    # measured RAW unweighted losses, which never reflected whatever weights
    # were actually set on SafeSpeechDisruptionLoss, silently showing the same
    # (pre-fix) imbalance no matter what weights you configured.
    from safespeech_losses import compute_pivotal_disruption_loss, compute_kl_to_noise, compute_l1_to_noise

    print(f"\n[diagnostic] Using configured weights: lambda_mel={disruption_loss_fn.lambda_mel}, "
          f"weight_l1={disruption_loss_fn.weight_l1}, weight_kl={disruption_loss_fn.weight_kl}")

    sub_loss_grad_norms = {}
    for sub_name in ["pivotal_disruption", "kl_to_noise", "l1_to_noise"]:
        for p_ in trainable_params:
            if p_.grad is not None:
                p_.grad = None

        out_sub = backbone.forward_full(clean_audio, message)
        recon_wm_sub = out_sub["recon_wm"]
        cloned_output_sub = surrogate.clone_voice(recon_wm_sub, text=args.surrogate_text)

        torch.manual_seed(999)  # fixed noise target across sub-components for a fair comparison
        random_noise = torch.randn_like(cloned_output_sub)

        if sub_name == "pivotal_disruption":
            raw_loss = compute_pivotal_disruption_loss(disruption_loss_fn.mel_fn, clean_audio, cloned_output_sub)
            sub_loss = -disruption_loss_fn.lambda_mel * raw_loss
        elif sub_name == "kl_to_noise":
            raw_loss = compute_kl_to_noise(disruption_loss_fn.mel_fn, cloned_output_sub, random_noise)
            sub_loss = disruption_loss_fn.weight_kl * raw_loss
        else:
            raw_loss = compute_l1_to_noise(disruption_loss_fn.mel_fn, cloned_output_sub, random_noise)
            sub_loss = disruption_loss_fn.weight_l1 * raw_loss

        sub_loss.backward()
        grad_norm = lora_param_grad_norm(backbone)
        sub_loss_grad_norms[sub_name] = grad_norm
        print(f"[diagnostic] {sub_name}: raw_loss={raw_loss.item():.4f}  "
              f"weighted_loss={sub_loss.item():.4f}  LoRA grad norm (weighted)={grad_norm:.8f}")

    print(f"\n{'=' * 60}")
    print("DIAGNOSIS")
    print(f"{'=' * 60}")
    total_sub_grad = sum(sub_loss_grad_norms.values())
    for name, norm in sub_loss_grad_norms.items():
        pct = 100 * norm / total_sub_grad if total_sub_grad > 0 else 0
        print(f"  {name}: {pct:.1f}% of combined sub-component gradient magnitude")

    pivotal_share = sub_loss_grad_norms["pivotal_disruption"] / total_sub_grad if total_sub_grad > 0 else 0
    if pivotal_share < 0.10:
        print(f"\nSTILL IMBALANCED: pivotal_disruption (the term that actually drives "
              f"SIM) contributes only {pivotal_share*100:.1f}% of the WEIGHTED disruption "
              f"gradient magnitude, even with the configured weights above. FIX: lower "
              f"--disrupt_weight_kl further and/or raise --disrupt_lambda_mel, then re-run "
              f"this diagnostic again before committing to a long training run.")
    else:
        print(f"\npivotal_disruption contributes a meaningful share ({pivotal_share*100:.1f}%) of "
              f"the gradient -- the earlier hypothesis about noise-term dominance doesn't fully "
              f"explain the flat SIM results. Worth checking Adam's adaptive normalization "
              f"behavior next, or whether SIM itself is a noisy/insensitive metric at this "
              f"eval-set size (25 utterances).")

    # --- Pass 3: NEW sim-targeted loss (the fix) -- check its raw gradient
    # scale before committing to a long training run, since its scale (raw
    # cosine similarity, ~0-1) is completely different from the old mel-based
    # loss's scale (~100+), meaning lambda_disrupt_max=1.0 will behave very
    # differently in --disrupt_mode sim.
    from safespeech_losses import compute_sim_disruption_loss
    for p_ in trainable_params:
        if p_.grad is not None:
            p_.grad = None

    out3 = backbone.forward_full(clean_audio, message)
    recon_wm3 = out3["recon_wm"]
    cloned_output3 = surrogate.clone_voice(recon_wm3, text=args.surrogate_text)
    emb_clean = surrogate.compute_speaker_embedding(clean_audio)
    emb_cloned = surrogate.compute_speaker_embedding(cloned_output3)
    sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)
    sim_loss.backward()
    sim_grad_norm = lora_param_grad_norm(backbone)

    print(f"\n{'=' * 60}")
    print("SIM-TARGETED LOSS (--disrupt_mode sim) -- gradient check before a long run")
    print(f"{'=' * 60}")
    print(f"sim_disruption_loss: raw_loss={sim_loss.item():.4f}  LoRA grad norm={sim_grad_norm:.8f}")
    print(f"VoiceMark grad norm (for reference): {vm_grad_norm:.8f}")
    if sim_grad_norm == 0.0:
        print("WARNING - zero gradient. Something is broken in the embedding path "
              "(check compute_speaker_embedding is not being called under no_grad).")
    else:
        suggested_lambda = vm_grad_norm / sim_grad_norm
        print(f"Suggested starting --lambda_disrupt_max for --disrupt_mode sim: "
              f"~{suggested_lambda:.2f} (to roughly match VoiceMark's own gradient scale -- "
              f"start here, don't assume it's exactly right, re-check after a short burst).")


if __name__ == "__main__":
    main()
