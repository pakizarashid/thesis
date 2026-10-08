"""
src/eval/composability_gen_shaped_audio.py   (2026-10-08)  -- SHAPED-NOISE variant of composability_gen_protected_audio.py

WHY
---
audio_level_and_noise_diagnostic.py showed that the flat L-infinity PGD noise is a constant-amplitude floor: at eps 0.002 the noise is ~30 dB
below the speech in speech frames (good) but only ~0 dB below the recording's own signal in pauses (84% of pause frames closer than 10 dB),
and at eps 0.005 it is 7 dB LOUDER than the pauses. That is audible hiss between words. This script keeps everything else identical and replaces the
flat bound by a TIME-VARYING bound that follows the loudness of the audio:

    bound[t] = eps * clamp( env[t] ** alpha , floor , 1 )          env = 20 ms RMS envelope of the watermarked clip, divided by its maximum

alpha = 0  -> bound[t] = eps for all t  = the existing flat PGD (a built-in control: it must reproduce the flat results).
alpha = 1  -> bound follows the signal amplitude (constant local SNR); quiet parts get proportionally less noise.
alpha = 0.5 -> halfway.   floor = lowest allowed fraction of eps (default 0.03 = -30 dB) so pauses are quiet but not literally zero.

Everything else (surrogate, loss, lambda_wm, steps, watermark, seeds, dataset, file names) is unchanged, so the output folder plugs into the same
evaluation (notebook 01 eval_protected / 01b). Because the average noise energy is lower at the same eps, compare at equal QUALITY / equal
audibility, not at equal eps: sweep eps_max upward (e.g. 0.005, 0.01, 0.02).

PRE-REGISTERED DECISION RULE (written before any run, 2026-10-08)
  Success of the shaped variant = at some setting, ALL of: PESQ >= 1.70, STOI >= 0.85, SI-SNR >= 0 dB (rows 1a-1c of the success threshold),
  ASR <= 50% (row 2a) and attacker-wins <= 10% (row 3a), on n = 100, AND the pause-frame SNR of the noise from audio_level_and_noise_diagnostic.py
  is >= 15 dB, AND the blind listening check (gallery blind_test.html, >= 20 trials per setting, loudness matched) is <= 60% correct.
  Partial = ASR at PESQ >= 1.70 clearly below the flat 0.002 result (69%): a paired difference of >= 15 points. Otherwise: the shaped variant
  fails; report it as a negative result. One variant family (alpha in {0.5, 1}), no further redesign.

Usage:
    python src/eval/composability_gen_shaped_audio.py --checkpoint ./checkpoints/route2_scaleupaug_msgproc_r2/route2_final.pt --msgproc_lora_r 2 \
        --epsilon 0.01 --alpha 1.0 --floor 0.03 --n_steps 10 --lambda_wm 1.0 --n_utterances 100 --save_dir ./eps_sweep_route2/shaped_a1_eps0.01_audio
    add --diagnostic first (1 utterance, verbose, nothing written).
"""
import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disruption_pgd import build_backbone, random_message, detect_acc
from surrogate_vc import load_yourtts_surrogate
from safespeech_losses import compute_sim_disruption_loss
from voicemark_losses import compute_ldec
from librispeech import LibriSpeechSubset, collate_librispeech


def envelope_bound(ref, epsilon, alpha, floor, win=321):
    """Per-sample L-inf bound, same shape as `ref`. ref = the (fixed) watermarked waveform."""
    shape = ref.shape
    x2 = ref.detach().reshape(1, 1, -1).pow(2)
    k = torch.ones(1, 1, win, device=ref.device, dtype=ref.dtype) / win
    env = torch.sqrt(F.conv1d(x2, k, padding=win // 2) + 1e-12)[..., : x2.shape[-1]]
    env = env / env.max().clamp_min(1e-9)
    m = env.pow(alpha).clamp(min=floor, max=1.0) if alpha > 0 else torch.ones_like(env)
    return (epsilon * m).reshape(shape)


def pgd_perturb_shaped(backbone, surrogate, clean_audio, message, text, epsilon, step_size, n_steps, random_start,
                       lambda_wm=1.0, alpha=1.0, floor=0.03, verbose=False):
    """Identical to disruption_pgd.pgd_perturb except the L-inf ball is a per-sample vector `bound` and the step is scaled by bound/epsilon."""
    with torch.no_grad():
        out = backbone.forward_full(clean_audio, message)
        recon_wm = out["recon_wm"].detach()
        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()
    bound = envelope_bound(recon_wm, epsilon, alpha, floor)
    step_vec = step_size * (bound / epsilon)

    delta = ((torch.rand_like(recon_wm) * 2 - 1) * bound) if random_start else torch.zeros_like(recon_wm)
    delta = delta.detach().requires_grad_(True)

    for step in range(n_steps):
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)
        cloned_output = surrogate.clone_voice(perturbed, text=text)
        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)
        sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)
        with torch.backends.cudnn.flags(enabled=False):          # same cuDNN-RNN-in-eval-mode workaround as the original
            detect_feat = backbone.model.st_model.forward_feature(perturbed)
            _logits, chunk_logits = backbone.model.detector(detect_feat)
            wm_loss = compute_ldec(chunk_logits, message)
        grad_sim = torch.autograd.grad(sim_loss, delta, retain_graph=True, create_graph=False)[0]
        grad_wm = torch.autograd.grad(wm_loss, delta, retain_graph=False, create_graph=False)[0]
        grad = grad_sim + lambda_wm * grad_wm
        if verbose:
            print(f"  [pgd step {step}] sim_loss={sim_loss.item():.4f} wm_loss={wm_loss.item():.4f} |grad|={grad.norm().item():.4e} "
                  f"delta_linf={delta.abs().max().item():.6f} mean_bound={bound.mean().item():.6f}")
        if step == 0 and grad.norm().item() == 0.0:
            print("  [pgd_perturb_shaped] WARNING: zero gradient at step 0 -- do not trust this run.")
        with torch.no_grad():
            delta = delta - step_vec * grad.sign()
            delta = torch.max(torch.min(delta, bound), -bound)
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        perturbed_final = torch.clamp(recon_wm + delta, -1.0, 1.0)
    return recon_wm, perturbed_final, delta.detach()


def _write(save_dir, tag, wav):
    os.makedirs(save_dir, exist_ok=True)
    sf.write(os.path.join(save_dir, f"{tag}.wav"), wav.detach().cpu().reshape(-1).numpy(), 16000)


def _match_rms(x, ref, peak=0.99):
    y = x * (ref.pow(2).mean().sqrt() + 1e-9) / (x.pow(2).mean().sqrt() + 1e-9)
    pk = y.abs().max()
    return y * (peak / pk) if pk > peak else y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=0.01, help="PEAK bound (reached where the audio is loudest)")
    p.add_argument("--alpha", type=float, default=1.0, help="0 = flat (control), 1 = bound proportional to the signal envelope")
    p.add_argument("--floor", type=float, default=0.03, help="lowest bound as a fraction of --epsilon (0.03 = -30 dB)")
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--match_rms", action="store_true", help="scale both outputs to the original's RMS as the last step (ACC measured after)")
    args = p.parse_args()
    if not args.diagnostic and args.save_dir is None:
        p.error("--save_dir is required unless --diagnostic is set")

    step_size = args.step_size if args.step_size is not None else args.epsilon / 4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_ds = LibriSpeechSubset(root=args.data_root, n_speakers=args.n_speakers, utterances_per_speaker=args.utterances_per_speaker,
                                n_eval_speakers=args.n_eval_speakers, eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                                sample_rate=16000, crop_seconds=args.crop_seconds, split="eval")
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech, drop_last=False)
    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, include_ffn=False, capacity_lora_r=32,
                              msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    print("[main] Loading YourTTS surrogate (PGD objective only -- no cloning happens in this script)...")
    surrogate = load_yourtts_surrogate(device=device)

    if args.diagnostic:
        batch = next(iter(loader))
        clean = batch["waveform"].to(device)
        msg = random_message(16, clean.shape[0], device, seed=123)
        recon, pert, delta = pgd_perturb_shaped(backbone, surrogate, clean, msg, args.surrogate_text, args.epsilon, step_size, args.n_steps,
                                                args.random_start, args.lambda_wm, args.alpha, args.floor, verbose=True)
        b = envelope_bound(recon, args.epsilon, args.alpha, args.floor)
        print(f"\nbound: mean {b.mean().item():.6f}  min {b.min().item():.6f}  max {b.max().item():.6f}  (flat PGD would be {args.epsilon})")
        print(f"ACC unprotected {detect_acc(backbone, recon, msg):.4f}   ACC protected {detect_acc(backbone, pert, msg):.4f}")
        return

    accs_u, accs_p = [], []
    print(f"\nSHAPED PGD | eps_max={args.epsilon} alpha={args.alpha} floor={args.floor} n_steps={args.n_steps} lambda_wm={args.lambda_wm} "
          f"| n={args.n_utterances} -> {args.save_dir}")
    for i, batch in enumerate(loader):
        if i >= args.n_utterances:
            break
        clean = batch["waveform"].to(device)
        msg = random_message(16, clean.shape[0], device, seed=123 + i)
        recon, pert, _ = pgd_perturb_shaped(backbone, surrogate, clean, msg, args.surrogate_text, args.epsilon, step_size, args.n_steps,
                                            args.random_start, args.lambda_wm, args.alpha, args.floor)
        if args.match_rms:
            recon, pert = _match_rms(recon, clean), _match_rms(pert, clean)
        au, ap = detect_acc(backbone, recon, msg), detect_acc(backbone, pert, msg)
        accs_u.append(au); accs_p.append(ap)
        _write(args.save_dir, f"sample{i}_reference", clean[0])
        _write(args.save_dir, f"sample{i}_unprotected", recon[0])
        _write(args.save_dir, f"sample{i}_protected", pert[0])
        print(f"  [{i}] acc_unprotected={au:.4f}  acc_protected={ap:.4f}", flush=True)
        if i == 0 and au < 0.85:
            raise SystemExit(f"ABORT: acc_unprotected={au:.4f} on utterance 0 (expected ~0.99). Check --checkpoint / --msgproc_lora_r.")

    manifest = {"label": "composability_audio_gen_shaped", "checkpoint": args.checkpoint, "msgproc_lora_r": args.msgproc_lora_r,
                "pgd": {"epsilon_peak": args.epsilon, "alpha": args.alpha, "floor": args.floor, "step_size": step_size,
                        "n_steps": args.n_steps, "lambda_wm": args.lambda_wm, "match_rms": args.match_rms},
                "n_completed": len(accs_u), "sanity_acc_unprotected_mean": sum(accs_u) / len(accs_u),
                "sanity_acc_protected_mean": sum(accs_p) / len(accs_p)}
    with open(os.path.join(args.save_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDONE. n={len(accs_u)} triplets in {args.save_dir}   mean ACC unprotected {manifest['sanity_acc_unprotected_mean']:.4f}  "
          f"protected {manifest['sanity_acc_protected_mean']:.4f}")


if __name__ == "__main__":
    main()
