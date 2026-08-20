"""
src/eval/gradient_diagnostic_capacity.py

Capacity-experiment variant of gradient_diagnostic.py. The original script
can't be pointed at the train_stage2_capacity.py backbone as-is:
  1. It calls apply_lora_adapters(backbone, r=args.lora_r, alpha=args.lora_alpha)
     with no include_ffn/ffn_r -- it would silently build the plain
     attention-only backbone, not the capacity one you actually want to test.
  2. Its checkpoint loader (backbone.model.load_state_dict(ckpt[...], strict=False))
     raises on SHAPE-mismatched tensors (strict=False only tolerates missing/
     extra KEYS, not shape mismatches on matching keys) -- since capacity_lora_r
     changes msg_processor's attention rank too (see adapters.py's
     apply_lora_adapters docstring: ffn_r overrides r for the WHOLE target,
     not just its FFN modules), loading a rank-8 Stage 1 checkpoint into a
     rank-32 msg_processor would crash here.
  3. lora_param_grad_norm() pools ALL LoRA params into one combined norm --
     it can't tell you whether the NEW feedforward parameters specifically
     are receiving meaningful gradient, which is the actual question this
     experiment needs answered before spending 30 epochs of compute.

This script fixes all three: builds the exact backbone train_stage2_capacity.py
trains, loads the Stage 1 checkpoint tolerantly (reusing
load_stage1_checkpoint_partial), and reports gradient norms broken down by
group (msg_processor attention / msg_processor feedforward / detector
attention) so you can see directly whether the feedforward capacity is doing
anything before committing to a long run -- same "measure before trusting"
discipline that caught the original 1000x kl_to_noise/pivotal imbalance.

WHAT TO LOOK FOR:
  - msg_processor_ffn's share of the total gradient norm should be
    NON-TRIVIAL (a rough rule of thumb: >5-10%, analogous to the >10% bar
    STAGE2_WRITEUP.md's diagnosis already uses for pivotal_disruption's
    share). If it's ~0%, the new capacity isn't receiving useful gradient
    at all and a 30-epoch run would just be spending compute to confirm
    what this 2-minute check already told you -- fix the lambda/weighting
    first (same lesson as the original bug), or reconsider the hypothesis.
  - Compare msg_processor_ffn's norm against msg_processor_attn's norm
    specifically -- if the new FFN capacity's gradient is many orders of
    magnitude smaller than the existing attention capacity's, Adam's
    per-parameter adaptive scaling will still let it move, but slowly;
    worth noting either way in your writeup.

Usage (mirrors train_stage2_capacity.py's own arguments so the backbone
you diagnose is EXACTLY the backbone you'd go on to train):
    python src/eval/gradient_diagnostic_capacity.py \\
        --stage1_checkpoint ./checkpoints/stage1_full_recalibrated_v3/recalibrated_final.pt \\
        --capacity_lora_r 32
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backbone import VoiceMarkBackbone, VoiceMarkDiscriminator
from adapters import apply_lora_adapters
from surrogate_vc import load_yourtts_surrogate
from voicemark_losses import VoiceMarkLosses
from safespeech_losses import (
    SafeSpeechDisruptionLoss, compute_pivotal_disruption_loss,
    compute_kl_to_noise, compute_l1_to_noise, compute_sim_disruption_loss,
)
from librispeech import LibriSpeechSubset, collate_librispeech
from torch.utils.data import DataLoader
from train_stage2_capacity import load_stage1_checkpoint_partial


def grad_norms_by_group(backbone) -> dict:
    """
    L2 norm of gradients, broken down into:
      msg_processor_attn / msg_processor_ffn / detector_attn / detector_ffn
    Identified purely from parameter name substrings -- 'in_proj_lora'/
    'out_proj_lora' = attention (LoRAMultiheadAttentionWrapper), 'linear_lora'
    = feedforward (LoRALinearWrapper, added for this experiment). See
    adapters.py's LoRALinearWrapper docstring for why it's named this way.
    """
    sums = {"msg_processor_attn": 0.0, "msg_processor_ffn": 0.0,
            "detector_attn": 0.0, "detector_ffn": 0.0, "other": 0.0}
    for name, p in backbone.model.named_parameters():
        if p.grad is None or "_lora." not in name:
            continue
        sq = p.grad.detach().norm(2).item() ** 2
        target = "msg_processor" if name.startswith("msg_processor") else (
            "detector" if name.startswith("detector") else None)
        kind = "ffn" if "linear_lora" in name else (
            "attn" if ("in_proj_lora" in name or "out_proj_lora" in name) else None)
        if target and kind:
            sums[f"{target}_{kind}"] += sq
        else:
            sums["other"] += sq
    return {k: v ** 0.5 for k, v in sums.items()}


def print_breakdown(label: str, norms: dict):
    total = sum(v ** 2 for v in norms.values()) ** 0.5
    print(f"  [{label}] total LoRA grad norm = {total:.8f}")
    for k, v in norms.items():
        if v == 0 and k == "other":
            continue
        pct = 100 * v / total if total > 0 else 0
        flag = "  <-- watch this one" if k == "msg_processor_ffn" else ""
        print(f"      {k:20s} = {v:.8f}  ({pct:5.1f}% of total){flag}")


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
    p.add_argument("--lora_r", type=int, default=8, help="detector's rank -- unchanged from canonical.")
    p.add_argument("--capacity_lora_r", type=int, default=32,
                    help="msg_processor's rank (attention AND the new feedforward LoRA) -- "
                         "match whatever you pass to train_stage2_capacity.py.")
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diagnostic] Using device: {device}")

    backbone = VoiceMarkBackbone()
    apply_lora_adapters(
        backbone, r=args.lora_r, alpha=args.lora_alpha,
        targets=("msg_processor", "detector"),
        include_ffn=True, ffn_r=args.capacity_lora_r, ffn_targets=("msg_processor",),
    )
    load_stage1_checkpoint_partial(backbone, args.stage1_checkpoint)

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
    trainable_params = [p_ for p_ in backbone.model.parameters() if p_.requires_grad]
    n_trainable = sum(p_.numel() for p_ in trainable_params)
    print(f"[diagnostic] Total trainable params in this backbone: {n_trainable:,} "
          f"(canonical Stage 2 was 294,912)")

    def zero_grads():
        for p_ in trainable_params:
            if p_.grad is not None:
                p_.grad = None

    # --- Pass 1: VoiceMark losses only (sanity baseline -- traceability path) ---
    zero_grads()
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
    print(f"\n[diagnostic] VoiceMark losses total = {vm_loss_dict['total'].item():.4f}")
    vm_norms = grad_norms_by_group(backbone)
    print_breakdown("VoiceMark losses", vm_norms)
    vm_grad_norm_total = sum(v ** 2 for v in vm_norms.values()) ** 0.5

    # --- Pass 2: mel-mode disruption sub-components (only relevant if you plan
    # to run --disrupt_mode mel; skip mentally if you're only using sim mode,
    # but cheap enough to always check) ---
    print(f"\n[diagnostic] Using configured weights: lambda_mel={disruption_loss_fn.lambda_mel}, "
          f"weight_l1={disruption_loss_fn.weight_l1}, weight_kl={disruption_loss_fn.weight_kl}")
    for sub_name in ["pivotal_disruption", "kl_to_noise", "l1_to_noise"]:
        zero_grads()
        out_sub = backbone.forward_full(clean_audio, message)
        recon_wm_sub = out_sub["recon_wm"]
        cloned_output_sub = surrogate.clone_voice(recon_wm_sub, text=args.surrogate_text)
        torch.manual_seed(999)
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
        print(f"\n[diagnostic] {sub_name}: raw_loss={raw_loss.item():.4f} weighted_loss={sub_loss.item():.4f}")
        print_breakdown(sub_name, grad_norms_by_group(backbone))

    # --- Pass 3: sim-mode disruption (the default/recommended mode) ---
    zero_grads()
    out3 = backbone.forward_full(clean_audio, message)
    recon_wm3 = out3["recon_wm"]
    cloned_output3 = surrogate.clone_voice(recon_wm3, text=args.surrogate_text)
    emb_clean = surrogate.compute_speaker_embedding(clean_audio)
    emb_cloned = surrogate.compute_speaker_embedding(cloned_output3)
    sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)
    sim_loss.backward()
    sim_norms = grad_norms_by_group(backbone)
    sim_grad_norm_total = sum(v ** 2 for v in sim_norms.values()) ** 0.5

    print(f"\n{'=' * 70}")
    print("SIM-TARGETED LOSS (--disrupt_mode sim, the default) -- gradient check")
    print(f"{'=' * 70}")
    print(f"sim_disruption_loss: raw_loss={sim_loss.item():.4f}")
    print_breakdown("sim_disruption", sim_norms)

    ffn_share = sim_norms["msg_processor_ffn"] / sim_grad_norm_total if sim_grad_norm_total > 0 else 0
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS")
    print(f"{'=' * 70}")
    if sim_grad_norm_total == 0.0:
        print("WARNING - zero gradient overall. Something is broken in the embedding path "
              "(check compute_speaker_embedding is not being called under no_grad).")
    elif ffn_share < 0.05:
        print(f"msg_processor's NEW feedforward capacity is receiving essentially no gradient "
              f"from the disruption loss ({ffn_share*100:.1f}% of the sim-mode total). A long "
              f"training run is unlikely to use this capacity meaningfully as-is -- before "
              f"spending 30 epochs, consider: (a) whether lambda_disrupt_max needs raising "
              f"further specifically to engage this pathway, (b) whether the feedforward "
              f"blocks are simply not on the dominant gradient path from recon_wm to the "
              f"surrogate's speaker embedding (plausible if the attention layers dominate "
              f"how msg_processor shapes its output), which would itself be a legitimate, "
              f"specific finding about WHERE capacity would need to go, not just whether more "
              f"of it helps.")
    else:
        print(f"msg_processor's new feedforward capacity IS receiving a meaningful share "
              f"({ffn_share*100:.1f}%) of the sim-mode disruption gradient -- worth proceeding "
              f"to a real training run. Suggested starting --lambda_disrupt_max: "
              f"~{(vm_grad_norm_total / sim_grad_norm_total):.2f} (matches VoiceMark's own "
              f"gradient scale, same derivation as the original diagnostic -- re-check after "
              f"a short calibration burst, don't assume it's exactly right).")


if __name__ == "__main__":
    main()
