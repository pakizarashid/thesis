"""
src/eval/protect_prompts_a0.py  (anti-cloning composability, surrogate comparison, 2026-10-26)

Adds the earlier control A0 as a fourth arm next to protect_prompts_v2.py's wm / rs / q2: the perturbation optimised
through the YourTTS surrogate instead of F5-TTS. A0 exactly as registered in the earlier runs
(pre-registration-hexternal-2026-10-23.md, "Control A0"): flat L-inf 0.002, 10 sign-PGD steps, step 0.0005,
lambda_wm 1, YourTTS similarity loss only (no ECAPA term), PGD random start and YourTTS sampling seeded with 5000 + i,
PGD sentence = the fixed sentence. disruption_pgd.pgd_perturb is called unchanged.

Same clips, crop and messages as protect_prompts_v2.py (its manifest is reused: the clean-prompt hash of every clip
is checked against it), written as 16-bit FLAC to OUT/a0/{i}.flac; source ACC and quality are measured on the saved
file. Needs the coqui-tts environment (YourTTS). Per-clip records go to OUT/records_a0/{i}.json, and --merge writes
OUT/a0_manifest.json.

    python src/eval/protect_prompts_a0.py --checkpoint CKPT --out_dir results/anticlone/prompts/RQA DATA_ARGS --start 0 --stop 50
    python src/eval/protect_prompts_a0.py ... --merge
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "models", "data", "losses"):
    sys.path.insert(0, os.path.join(HERE, "..", sub) if sub else os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import clone_eval_v2 as ce  # noqa: E402
import cloner_watermark_eval as cwe  # noqa: E402
from protect_prompts_v2 import quality, read_audio, wav_sha, write_flac  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out_dir", required=True, help="the protect_prompts_v2.py output dir of the same checkpoint")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=0.002)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--noise_seed_base", type=int, default=5000)
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--pgd_text", default=ce.FIXED_TEXT)
    p.add_argument("--dataset", choices=("librispeech", "vctk"), default="librispeech")
    p.add_argument("--data_root", default="./data/librispeech")
    p.add_argument("--libri_subset", default="train-clean-100")
    p.add_argument("--vctk_root", default=None)
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--max_full_seconds", type=float, default=20.0)
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.load(open(os.path.join(args.out_dir, "manifest.json")))
    assert man["complete"], "run protect_prompts_v2.py (and its --merge) for this checkpoint first"
    ck_sha = ce.sha256(args.checkpoint)
    assert man["settings"]["checkpoint_sha256"] == ck_sha, "--checkpoint differs from the one the other arms were made with"
    base = {r["index"]: r for r in man["records"]}
    n_clips = min(args.n_utterances, man["n_clips"])
    rec_dir = os.path.join(args.out_dir, "records_a0")

    if args.merge:
        recs = [json.load(open(os.path.join(rec_dir, f"{i}.json"))) for i in range(n_clips)
                if os.path.exists(os.path.join(rec_dir, f"{i}.json"))]
        missing = sorted(set(range(n_clips)) - {r["index"] for r in recs})
        summ = {k: float(np.mean([r["a0"][k] for r in recs])) for k in ("acc_source", "pesq", "stoi", "si_snr", "snr_db")} if recs else {}
        out = {"script": "protect_prompts_a0", "settings": {k: v for k, v in vars(args).items() if k not in ("start", "stop", "merge")},
               "checkpoint_sha256": ck_sha, "n_clips": n_clips, "missing": missing, "complete": not missing,
               "summary": {"a0": summ}, "records": sorted(recs, key=lambda r: r["index"])}
        json.dump(out, open(os.path.join(args.out_dir, "a0_manifest.json"), "w"), indent=1)
        print(json.dumps({"n_done": len(recs), "missing": missing[:20], "summary": summ}, indent=1))
        return

    from disruption_pgd import pgd_perturb
    from surrogate_vc import load_yourtts_surrogate
    ds = ce.load_dataset(args, crop=args.max_full_seconds)
    backbone = cwe.build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    surrogate = load_yourtts_surrogate(device=device)
    lo, hi = args.start, min(args.stop if args.stop is not None else n_clips, n_clips)
    os.makedirs(rec_dir, exist_ok=True)
    t0, n_new = time.time(), 0
    for i in range(lo, hi):
        rp, fp = os.path.join(rec_dir, f"{i}.json"), os.path.join(args.out_dir, "a0", f"{i}.flac")
        if os.path.exists(rp) and os.path.exists(fp):
            continue
        it = ds[i]
        clean = ce.make_prompt(ce.trim_padding(it["waveform"].reshape(-1)), "3", args.max_full_seconds).to(device).reshape(1, 1, -1)
        assert wav_sha(clean) == base[i]["clean_sha"], f"clip {i}: clean prompt differs from the other arms"
        msg = ce.message_bits(args.message_seed_base + i, device)
        seed = args.noise_seed_base + i
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)       # YourTTS samples noise on every clone call
        _wm, a0, _d = pgd_perturb(backbone, surrogate, clean, msg, args.pgd_text, args.epsilon, args.epsilon / 4.0,
                                  args.n_steps, True, lambda_wm=args.lambda_wm, noise_seed=seed)
        write_flac(fp, a0)
        back = read_audio(fp, device)
        dec = np.array(ce.decode_bits(backbone, back))
        rec = {"index": i, "utterance_id": base[i]["utterance_id"], "clean_sha": base[i]["clean_sha"],
               "a0": {"acc_source": float((dec == np.array(msg[0].tolist())).mean()), **quality(clean, back)}}
        json.dump(rec, open(rp, "w"), indent=1)
        n_new += 1
        el = time.time() - t0
        print(f"[{i}] a0 acc {rec['a0']['acc_source']:.3f} pesq {rec['a0']['pesq']:.2f} | {el / n_new:.0f} s/clip, "
              f"~{el / n_new * (hi - i - 1) / 60:.0f} min left", flush=True)
    print(f"DONE clips {lo}..{hi - 1}")


if __name__ == "__main__":
    main()
