"""
src/eval/quality_paired_ckpts.py

Paired, no-cloning evaluation of several watermark checkpoints on the SAME
held-out clips (2026-10-24). For each checkpoint and each clip:

  - PESQ (wideband, 16 kHz), STOI and SI-SNR of the watermarked audio against
    the clean original (the project's audio-quality bars: 1.70 / 0.85 / 0 dB)
  - the same three metrics for the codec reconstruction WITHOUT a watermark
    (the codec-only row; never measured with a result file before)
  - bitwise watermark ACC on the clean watermarked audio
  - optionally (--aug_eval) bitwise ACC after each VoiceMark augmentation type
    separately (mask, shuffle, replace, encodec, perturb), with a fixed seed so
    every checkpoint sees identical edits. This is the direct check that a
    detector trained with src/data/augment_voicemark.py is robust to those
    edits, and that an older checkpoint is not.

The first checkpoint is the control. Every other checkpoint is compared with it
clip by clip (mean difference, paired t-test, Wilcoxon). Per-clip values are
saved, so any later test can be re-run from the JSON.

Message for clip i: 16 bits drawn with seed --message_seed_base + i (project
convention 123 + i), identical across checkpoints.

Usage:
    python src/eval/quality_paired_ckpts.py \\
        --ckpt old=checkpoints/stage1_scaleup_aug/stage1_epoch29.pt \\
        --ckpt new=checkpoints/stage1_vmaug/stage1_epoch29.pt \\
        --ckpt pretrained=none \\
        --aug_eval --output results/results_quality_paired_stage1_vmaug.json

    A checkpoint spec is NAME=PATH, NAME=PATH:MSGPROC_RANK, or NAME=none for
    pretrained VoiceMark (LoRA at zero init).
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("models", "data", "losses"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

from backbone import VoiceMarkBackbone  # noqa: E402
from adapters import apply_lora_adapters  # noqa: E402
from librispeech import LibriSpeechSubset  # noqa: E402


def parse_spec(spec):
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise ValueError(f"bad --ckpt {spec!r}; use NAME=PATH, NAME=PATH:RANK or NAME=none")
    rank = None
    if rest != "none" and ":" in rest and rest.rsplit(":", 1)[1].isdigit():
        rest, r = rest.rsplit(":", 1)
        rank = int(r)
    return name, (None if rest == "none" else rest), rank


def load_backbone(path, lora_r, lora_alpha, msgproc_rank, device):
    backbone = VoiceMarkBackbone()
    target_ranks = {"msg_processor": msgproc_rank} if msgproc_rank else None
    apply_lora_adapters(backbone, r=lora_r, alpha=lora_alpha, target_ranks=target_ranks)
    if path is not None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["lora_state_dict"]
        own = backbone.model.state_dict()
        bad = [k for k in state if k in own and state[k].shape != own[k].shape]
        if bad:
            raise RuntimeError(f"{path}: {len(bad)} LoRA tensors have the wrong shape "
                               f"(e.g. {bad[0]}); pass the matching rank as NAME=PATH:RANK")
        missing = [k for k in own if "_lora." in k and k not in state]
        backbone.model.load_state_dict(state, strict=False)
        print(f"[load] {path}: {len(state)} LoRA tensors loaded"
              + (f", {len(missing)} LoRA tensors NOT in checkpoint (stay at zero init)" if missing else ""))
    backbone.model.to(device).eval()
    return backbone


def bit_acc(chunk_logits, message, nchunk=4):
    pred = torch.argmax(chunk_logits, dim=-1)
    correct = total = 0
    for i in range(pred.shape[1]):
        for b in range(nchunk):
            correct += (((pred[:, i] >> b) & 1) == message[:, i * nchunk + b]).sum().item()
            total += pred.shape[0]
    return correct / total


def detect(backbone, audio, message):
    feat = backbone.model.st_model.forward_feature(audio)
    _, chunk_logits = backbone.model.detector(feat)
    return bit_acc(chunk_logits, message)


def si_snr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    noise = est - proj
    return 10 * np.log10((np.dot(proj, proj) + eps) / (np.dot(noise, noise) + eps))


def quality(est, ref, sr=16000):
    from pesq import pesq
    from pystoi import stoi
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    try:
        p = float(pesq(sr, ref, est, "wb"))
    except Exception as e:  # PESQ fails on near-silent clips; record, do not crash
        print(f"    PESQ failed ({e}); recorded as NaN")
        p = float("nan")
    return p, float(stoi(ref, est, sr, extended=False)), float(si_snr(est, ref))


def paired(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    out = {"n": int(ok.sum()), "mean_control": float(a.mean()), "mean_other": float(b.mean()),
           "mean_diff_other_minus_control": float(d.mean())}
    if len(d) > 1 and np.any(d != 0):
        out["t_p"] = float(stats.ttest_rel(b, a).pvalue)
        out["wilcoxon_p"] = float(stats.wilcoxon(b, a).pvalue)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="NAME=PATH[:MSGPROC_RANK] or NAME=none. First one is the control.")
    p.add_argument("--output", required=True)
    p.add_argument("--aug_eval", action="store_true", help="Also ACC per VoiceMark augmentation type.")
    p.add_argument("--aug_seed", type=int, default=2024)
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--dataset", choices=("librispeech", "vctk"), default="librispeech")
    p.add_argument("--vctk_root", type=str,
                   default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--data_root", default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = [parse_spec(s) for s in args.ckpt]

    if args.dataset == "librispeech":
        ds = LibriSpeechSubset(root=args.data_root, n_speakers=args.n_speakers,
                               utterances_per_speaker=args.utterances_per_speaker,
                               n_eval_speakers=args.n_eval_speakers,
                               eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                               sample_rate=16000, crop_seconds=args.crop_seconds, split="eval")
    else:
        from vctk import VCTKSubset  # same constructor as xtts_transfer_eval.py uses
        ds = VCTKSubset(root=args.vctk_root, n_speakers=args.n_speakers,
                        utterances_per_speaker=args.utterances_per_speaker,
                        n_eval_speakers=args.n_eval_speakers,
                        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval")
    n = len(ds)
    print(f"[data] {n} held-out clips")

    augmenters = {}
    if args.aug_eval:
        from augment_voicemark import VoiceMarkAugmenter, ALL_AUGMENTATIONS
        for a in ALL_AUGMENTATIONS:
            augmenters[a] = VoiceMarkAugmenter(16000, mode="one", augmentations=[a], p_identity=0.0)

    per = {}
    codec_only = {"pesq": [], "stoi": [], "si_snr": []}
    for ci, (name, path, rank) in enumerate(specs):
        print(f"\n=== {name} ({path or 'pretrained VoiceMark'}) ===")
        backbone = load_backbone(path, args.lora_r, args.lora_alpha, rank, device)
        rec = {"pesq": [], "stoi": [], "si_snr": [], "acc_clean": [], "clip_ids": []}
        for a in augmenters:
            rec[f"acc_{a}"] = []
        for i in range(n):
            item = ds[i]
            clean = item["waveform"].unsqueeze(0).to(device)  # [1, 1, T]
            gen = torch.Generator(device="cpu").manual_seed(args.message_seed_base + i)
            msg = torch.randint(0, 2, (1, 16), generator=gen).to(device)
            with torch.no_grad():
                out = backbone.forward_full(clean, msg)
                wm = out["recon_wm"]
                rec["acc_clean"].append(detect(backbone, wm, msg))
                for a, aug in augmenters.items():
                    torch.manual_seed(args.aug_seed + i)
                    random.seed(args.aug_seed + i)
                    rec[f"acc_{a}"].append(detect(backbone, aug(wm, clean)["audio"], msg))
            c = clean[0, 0].cpu().numpy()
            q = quality(wm[0, 0].cpu().numpy(), c)
            rec["pesq"].append(q[0])
            rec["stoi"].append(q[1])
            rec["si_snr"].append(q[2])
            rec["clip_ids"].append(item.get("utterance_id", i))
            if ci == 0:  # codec-only row: same for every checkpoint (LoRA does not touch the codec)
                qc = quality(out["recon"][0, 0].cpu().numpy(), c)
                codec_only["pesq"].append(qc[0])
                codec_only["stoi"].append(qc[1])
                codec_only["si_snr"].append(qc[2])
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{n} clips")
        per[name] = rec
        means = {k: float(np.nanmean(v)) for k, v in rec.items() if k != "clip_ids"}
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
        del backbone
        torch.cuda.empty_cache()

    control = specs[0][0]
    comparisons = {}
    for name, _, _ in specs[1:]:
        comparisons[f"{name}_vs_{control}"] = {
            k: paired(per[control][k], per[name][k]) for k in per[control] if k != "clip_ids"
        }

    summary = {name: {k: float(np.nanmean(v)) for k, v in rec.items() if k != "clip_ids"}
               for name, rec in per.items()}
    summary["codec_only_no_watermark"] = {k: float(np.nanmean(v)) for k, v in codec_only.items()}
    result = {"args": vars(args), "summary": summary, "paired_vs_first": comparisons,
              "per_clip": per, "codec_only_per_clip": codec_only}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Summary (means) ===")
    cols = sorted({k for v in summary.values() for k in v})
    print("checkpoint".ljust(26) + "".join(c.rjust(14) for c in cols))
    for name, v in summary.items():
        print(name.ljust(26) + "".join((f"{v[c]:.4f}" if c in v else "-").rjust(14) for c in cols))
    for cmp_name, d in comparisons.items():
        print(f"\n{cmp_name}")
        for k, s in d.items():
            ps = "".join(f"  {pk}={s[pk]:.2g}" for pk in ("t_p", "wilcoxon_p") if pk in s)
            print(f"  {k:14s} diff={s['mean_diff_other_minus_control']:+.4f}{ps}")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
