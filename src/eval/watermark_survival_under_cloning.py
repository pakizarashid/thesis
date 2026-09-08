"""
src/eval/watermark_survival_under_cloning.py

Closes a real, previously-unmeasured gap: every existing script in this repo
either (a) measures whether the watermark survives OUR OWN perturbation
(disruption_pgd.py's detection_acc_before/after, both computed on recon_wm /
perturbed_final -- the pre-clone audio) or (b) measures whether cloning
degrades the CLONE'S similarity to the target speaker (disruption_pgd.py's
sim_before/sim_after). Neither ever runs the watermark detector on the
actual cloned output. That's the specific test VoiceMark's own paper does
(Table 1: watermark ACC on audio AFTER passing through CosyVoice/F5-TTS/
MaskGCT) -- and it's a different, more direct question than either existing
metric answers: if someone actually clones this protected speaker's voice
via a real zero-shot TTS system, does VoiceMark's watermark still show up
in THAT output at all?

This matters structurally, not just as a missing number. YourTTS's
clone_voice() (see surrogate_vc.py) is TEXT-conditioned zero-shot synthesis:
it extracts a speaker embedding from the input audio and generates a BRAND
NEW waveform from text via its own flow/vocoder pipeline. The original
waveform -- and whatever fine-grained signal pattern VoiceMark's embedder
wrote into it -- is not reused at all; only a compressed identity embedding
survives into the new synthesis. Whether a waveform-level watermark can
possibly survive that pipeline is an open, genuinely uncertain question --
not something to assume either way from the SIM numbers already collected,
which never touch the watermark detector. This script measures it directly
instead of assuming.

THREE numbers per run, so the result is interpretable either way:
  - detection_acc_on_source: sanity check, watermark ACC on the pre-clone
    audio itself (recon_wm or perturbed_final) -- should match existing
    disruption_pgd.py numbers; if it doesn't, something upstream broke.
  - detection_acc_on_clone_of_unprotected: the VoiceMark-Table-1-equivalent
    metric -- watermark ACC on YourTTS's clone of the WATERMARKED-BUT-NOT-
    DISRUPTED audio. This is the number that answers "does our traceability
    claim actually survive a real zero-shot cloning attack," independent of
    the disruption mechanism entirely.
  - detection_acc_on_clone_of_protected (only if --epsilon > 0): watermark
    ACC on YourTTS's clone of the PGD-protected audio -- the real end-to-end
    dual-defense number: does the watermark survive an attacker who clones
    the protected audio anyway (SIM already shows the clone is degraded;
    this checks whether it's ALSO traceable if that degraded clone is used).

Usage:
    # Sanity/no-protection run first -- does the watermark survive real
    # zero-shot cloning at all, before any disruption is involved:
    python src/eval/watermark_survival_under_cloning.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --output results/results_clonewm_unprotected_run1.json

    # Full dual-defense run, at the established operating point:
    python src/eval/watermark_survival_under_cloning.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --epsilon 0.002 --n_steps 10 --lambda_wm 1.0 \\
        --output results/results_clonewm_protected_eps002_run1.json
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disruption_pgd import build_backbone, random_message, detect_acc, pgd_perturb, compute_sim
from surrogate_vc import load_yourtts_surrogate
from librispeech import LibriSpeechSubset, collate_librispeech
from vctk import VCTKSubset, collate_vctk
from libritts import LibriTTSSubset, collate_libritts
from torch.utils.data import DataLoader


def run_eval(backbone, surrogate, eval_loader, device, text: str,
             epsilon: float, step_size: float, n_steps: int, lambda_wm: float,
             seed: int = 123) -> dict:
    keys = ["detection_acc_on_source", "detection_acc_on_clone_of_unprotected", "sim_of_unprotected_clone"]
    if epsilon > 0:
        keys += ["detection_acc_on_clone_of_protected", "sim_of_protected_clone"]
    metrics = {k: [] for k in keys}

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=seed + batch_idx)

        with torch.no_grad():
            out = backbone.forward_full(clean_audio, message)
            recon_wm = out["recon_wm"].detach()

        # Unprotected path: clone the watermarked-but-undisrupted audio directly.
        # This is the number VoiceMark's own paper reports for CosyVoice/F5-TTS/
        # MaskGCT -- here reproduced against a real zero-shot model (YourTTS) on
        # THIS project's own checkpoint, for the first time in this repo.
        with torch.no_grad():
            cloned_unprotected = surrogate.clone_voice(recon_wm, text=text)
        acc_source = detect_acc(backbone, recon_wm, message)
        acc_clone_unprotected = detect_acc(backbone, cloned_unprotected, message)
        sim_unprotected = compute_sim(surrogate, clean_audio, cloned_unprotected)

        metrics["detection_acc_on_source"].append(acc_source)
        metrics["detection_acc_on_clone_of_unprotected"].append(acc_clone_unprotected)
        metrics["sim_of_unprotected_clone"].append(sim_unprotected)

        log_line = (f"[batch {batch_idx}] acc_source={acc_source:.4f} "
                    f"acc_clone(unprotected)={acc_clone_unprotected:.4f} "
                    f"sim(unprotected)={sim_unprotected:.4f}")

        if epsilon > 0:
            _, perturbed_final, _ = pgd_perturb(
                backbone, surrogate, clean_audio, message, text,
                epsilon, step_size, n_steps, random_start=True, lambda_wm=lambda_wm,
            )
            with torch.no_grad():
                cloned_protected = surrogate.clone_voice(perturbed_final, text=text)
            acc_clone_protected = detect_acc(backbone, cloned_protected, message)
            sim_protected = compute_sim(surrogate, clean_audio, cloned_protected)

            metrics["detection_acc_on_clone_of_protected"].append(acc_clone_protected)
            metrics["sim_of_protected_clone"].append(sim_protected)
            log_line += (f" | acc_clone(protected)={acc_clone_protected:.4f} "
                         f"sim(protected)={sim_protected:.4f}")

        print(log_line)

    return {k: sum(v) / len(v) for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epsilon", type=float, default=0.0,
                    help="0.0 (default) = unprotected-only run: measures whether the "
                         "watermark survives real zero-shot cloning with NO disruption "
                         "involved at all. Set > 0 to also test the full dual-defense "
                         "path (protected audio, then cloned).")
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--dataset", type=str, default="librispeech", choices=["librispeech", "vctk", "libritts"])
    p.add_argument("--vctk_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--libritts_root", type=str,
                    default="/kaggle/input/datasets/pratt3000/libritts/LibriTTS")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    args = p.parse_args()

    step_size = args.step_size if args.step_size is not None else args.epsilon / 4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "librispeech":
        eval_ds = LibriSpeechSubset(
            root=args.data_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_librispeech
    elif args.dataset == "vctk":
        eval_ds = VCTKSubset(
            root=args.vctk_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_vctk
    else:
        eval_ds = LibriTTSSubset(
            root=args.libritts_root, n_speakers=args.n_speakers,
            utterances_per_speaker=args.utterances_per_speaker,
            n_eval_speakers=args.n_eval_speakers,
            eval_utterances_per_speaker=args.eval_utterances_per_speaker,
            sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
        )
        collate_fn = collate_libritts

    print(f"[main] dataset={args.dataset} epsilon={args.epsilon} "
          f"(0.0 = unprotected-only) lambda_wm={args.lambda_wm}")
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=False)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                               include_ffn=False, capacity_lora_r=32)
    print("[main] Loading YourTTS surrogate (frozen)...")
    surrogate = load_yourtts_surrogate(device=device)

    means = run_eval(backbone, surrogate, eval_loader, device, args.surrogate_text,
                      args.epsilon, step_size, args.n_steps, args.lambda_wm)

    print("\n" + "=" * 60)
    print(f"Sanity check, ACC on pre-clone source audio: {means['detection_acc_on_source']:.4f}")
    print(f"ACC on real YourTTS clone of UNPROTECTED watermarked audio: "
          f"{means['detection_acc_on_clone_of_unprotected']:.4f}  <-- the VoiceMark-Table-1-equivalent number")
    if args.epsilon > 0:
        print(f"ACC on real YourTTS clone of PROTECTED (PGD-disrupted) audio: "
              f"{means['detection_acc_on_clone_of_protected']:.4f}  <-- the real end-to-end dual-defense number")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({
            "label": f"clonewm_eps{args.epsilon}_lwm{args.lambda_wm}"
                      + (f"_on_{args.checkpoint}" if args.checkpoint else "_on_baseline"),
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "pgd": {"epsilon": args.epsilon, "step_size": step_size,
                     "n_steps": args.n_steps, "lambda_wm": args.lambda_wm},
            "results": {**means, "n_trials": len(eval_ds)},
        }, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
