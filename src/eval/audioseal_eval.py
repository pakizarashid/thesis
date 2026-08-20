"""
src/eval/audioseal_eval.py

Reproduces AudioSeal as an actual, in-repo baseline -- resolving the gap
flagged during the repo audit: AudioSeal was listed as "already reproduced"
in the project plan but no code, checkpoint, or results file for it existed
anywhere in this repository. This script fills that gap using Meta's own
official `audioseal` pip package (not a from-scratch reimplementation --
AudioSeal's pretrained release is the correct thing to compare against, the
same way this project uses VoiceMark's own released checkpoint rather than
retraining it).

Mirrors audiopure_eval.py's exact structure and output schema (acc_before /
acc_after / acc_drop) on the SAME eval set (LibriSpeechSubset, same speakers/
utterances/crop_seconds arguments) so results/results_audioseal_audiopure.json
drops straight into aggregate_results.py's table next to VoiceMark's own
baseline/Stage1/Stage2/AudioPure numbers -- a genuine reactive-only-watermark
reference point, not a citation-only claim.

Install (once):
    pip install audioseal --break-system-packages

Usage (matches audiopure_eval.py's own usage pattern):
    python src/eval/audioseal_eval.py --output results_audioseal_audiopure.json
    python src/eval/audioseal_eval.py --output results_audioseal_clean.json --skip_purification

CONFIDENCE NOTE: AudioSeal's public API (AudioSeal.load_generator /
load_detector, get_watermark(), detect_watermark()) was confirmed from the
official facebookresearch/audioseal README as of writing this script, not
guessed -- but treat it the same way this project treats every other
third-party integration (VoiceMark, AudioPure, YourTTS): re-verify against
the installed package version's actual source/docstrings before trusting a
long run, since APIs do drift across releases.
"""

import os
import sys
import json
import argparse
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from librispeech import LibriSpeechSubset, collate_librispeech
from torch.utils.data import DataLoader


def build_audioseal(device: str, nbits_variant: str = "16bits"):
    """
    Loads AudioSeal's pretrained generator + detector. Unlike VoiceMark,
    AudioSeal's message is embedded via get_watermark(..., message=...) and
    ADDED to the waveform (watermark = model.get_watermark(wav); wm_audio =
    wav + watermark) -- a residual-additive scheme, structurally different
    from VoiceMark's RVQ-latent carrier. That's expected and is itself part
    of the comparison: AudioSeal is the project's REACTIVE-ONLY baseline
    (per the original plan), so a different, simpler embedding mechanism is
    exactly what "reactive-only, not survives-cloning-by-design" looks like.
    """
    from audioseal import AudioSeal

    generator = AudioSeal.load_generator(f"audioseal_wm_{nbits_variant}")
    detector = AudioSeal.load_detector(f"audioseal_detector_{nbits_variant}")
    generator.to(device).eval()
    detector.to(device).eval()
    return generator, detector


def build_audiopure_denoiser(repo_root: str, reverse_timestep: int = 25):
    """Identical to audiopure_eval.py's own loader -- kept local to avoid a
    cross-file import dependency that would break if that file's internals
    change; if you'd rather share one implementation, factor this out into
    a shared src/eval/_audiopure_common.py instead."""
    sys.path.insert(0, repo_root)
    from external.audiopure.diffusion_models.diffwave_ddpm import create_diffwave_model

    model_path = os.path.join(
        repo_root, "external/audiopure/diffusion_models/DiffWave_Unconditional/"
        "exp/ch256_T200_betaT0.02/logs/checkpoint/1000000.pkl"
    )
    config_path = os.path.join(
        repo_root, "external/audiopure/diffusion_models/DiffWave_Unconditional/config.json"
    )
    denoiser = create_diffwave_model(model_path=model_path, config_path=config_path,
                                      reverse_timestep=reverse_timestep)
    print(f"[build_audiopure_denoiser] Loaded DiffWave denoiser (reverse_timestep={reverse_timestep})")
    return denoiser


def bit_accuracy(pred_msg: torch.Tensor, true_msg: torch.Tensor) -> float:
    """pred_msg/true_msg: [batch, nbits] in {0,1}. Matches this project's
    other eval scripts' bitwise-ACC convention (not a whole-message exact-match
    metric), for direct numeric comparability with VoiceMark's ACC numbers."""
    return (pred_msg == true_msg).float().mean().item()


def run_eval(generator, detector, denoiser, eval_loader, device, seed: int = 123,
             skip_purification: bool = False) -> dict:
    acc_before_list, acc_after_list, presence_before_list, presence_after_list = [], [], [], []
    purify_times = []

    for batch_idx, batch in enumerate(eval_loader):
        clean_audio = batch["waveform"].to(device)
        gen = torch.Generator(device=device).manual_seed(seed + batch_idx)
        message = torch.randint(0, 2, (clean_audio.shape[0], 16), generator=gen, device=device)

        with torch.no_grad():
            watermark = generator.get_watermark(clean_audio, message=message)
            wm_audio = clean_audio + watermark

            presence_before, decoded_before = detector.detect_watermark(wm_audio)
            # detect_watermark's decoded message is soft (bit probabilities in
            # some versions) -- threshold at 0.5 to get a hard bit prediction,
            # matching the bitwise-ACC convention used throughout this project.
            decoded_before_hard = (decoded_before > 0.5).long()
            acc_before = bit_accuracy(decoded_before_hard, message)
            acc_before_list.append(acc_before)
            presence_before_list.append(float(presence_before if not torch.is_tensor(presence_before)
                                               else presence_before.mean().item()))

            if not skip_purification:
                t0 = time.time()
                purified = denoiser(wm_audio)
                purify_times.append(time.time() - t0)

                presence_after, decoded_after = detector.detect_watermark(purified)
                decoded_after_hard = (decoded_after > 0.5).long()
                acc_after = bit_accuracy(decoded_after_hard, message)
                acc_after_list.append(acc_after)
                presence_after_list.append(float(presence_after if not torch.is_tensor(presence_after)
                                                  else presence_after.mean().item()))

        log_line = f"  utterance {batch_idx}: acc_before={acc_before:.4f}"
        if not skip_purification:
            log_line += f" acc_after={acc_after:.4f} (purify_time={purify_times[-1]:.1f}s)"
        print(log_line)

    results = {
        "acc_before_mean": sum(acc_before_list) / len(acc_before_list),
        "acc_before_values": acc_before_list,
        "presence_before_mean": sum(presence_before_list) / len(presence_before_list),
    }
    if not skip_purification:
        results["acc_after_mean"] = sum(acc_after_list) / len(acc_after_list)
        results["acc_after_values"] = acc_after_list
        results["presence_after_mean"] = sum(presence_after_list) / len(presence_after_list)
        results["mean_purify_time_s"] = sum(purify_times) / len(purify_times)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--nbits_variant", type=str, default="16bits")
    p.add_argument("--reverse_timestep", type=int, default=25)
    p.add_argument("--skip_purification", action="store_true",
                    help="Just measure clean-condition ACC (sanity check / faster iteration), "
                         "skip loading AudioPure entirely.")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=10)
    p.add_argument("--utterances_per_speaker", type=int, default=10)
    p.add_argument("--n_eval_speakers", type=int, default=5)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_librispeech)

    print(f"\n{'=' * 60}\nEvaluating: AudioSeal ({args.nbits_variant}, reactive-only baseline)\n{'=' * 60}")

    generator, detector = build_audioseal(device, nbits_variant=args.nbits_variant)
    denoiser = None if args.skip_purification else build_audiopure_denoiser(args.repo_root, args.reverse_timestep)

    results = run_eval(generator, detector, denoiser, eval_loader, device,
                        skip_purification=args.skip_purification)

    print(f"\n{'=' * 60}")
    print(f"ACC before purification: {results['acc_before_mean']:.4f}")
    if not args.skip_purification:
        print(f"ACC after purification:  {results['acc_after_mean']:.4f}")
        print(f"ACC drop: {results['acc_before_mean'] - results['acc_after_mean']:.4f}")

    out_results = {"acc_before": results["acc_before_mean"]}
    if not args.skip_purification:
        out_results["acc_after"] = results["acc_after_mean"]
        out_results["acc_drop"] = results["acc_before_mean"] - results["acc_after_mean"]

    with open(args.output, "w") as f:
        json.dump({"label": f"audioseal_{args.nbits_variant}", "checkpoint": "pretrained (Meta release)",
                    "results": out_results}, f, indent=2)
    print(f"[main] Saved results to {args.output}")


if __name__ == "__main__":
    main()
