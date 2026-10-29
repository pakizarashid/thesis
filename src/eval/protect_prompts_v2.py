"""
src/eval/protect_prompts_v2.py  (anti-cloning composability, 2026-10-26)

Builds the protected 3 s prompts that clone_eval_v2.py --prompt_dir clones, on EXACTLY the clip set,
crop and messages clone_eval_v2.py uses (its load_dataset / trim_padding / make_prompt / message_bits are
imported, not copied), for one watermark checkpoint. Three arms per clip:

    wm : watermark only                                 recon_wm
    rs : random-sign noise in Q2's bound (control)      recon_wm + bound * sign(u),  u = Q2's own PGD random start
    q2 : the registered Q2 perturbation                 f5_whitebox_pgd.whitebox_pgd, arm "wens"
         (peak 0.015, envelope alpha 1, floor 0.003, 40 steps, step 0.0015, dis:wm ratio 1.732, lambda_wm 1,
          F5 K=6, CFG 2, PGD sentence = the fixed sentence, prompt transcript = faster-whisper base.en of the clean prompt,
          PGD start seed 5000 + i, F5 rollout seed 7000 + i)

rs has the same per-sample bound as q2 and sits at the bound everywhere (sign-PGD ends mostly at the bound), so it
matches q2's noise energy; it says whether q2's effect needs the optimisation or only the noise.

Every arm is written as 16-bit PCM FLAC (what an attacker downloads), and every number below (source ACC, PESQ,
STOI, SI-SNR against the clean prompt) is computed on the file as reloaded, so quantisation is inside the test.

Output layout (out_dir):
    wm/{i}.flac  rs/{i}.flac  q2/{i}.flac
    records/{i}.json             per clip (written as each clip finishes, so two processes can share out_dir)
    manifest.json                --merge: all records + settings, complete=True when every clip 0..n-1 is there

clone_eval_v2.py --prompt_dir out_dir/ARM checks, per clip, that the clean prompt hash in the record equals the
hash of its own clean 3 s prompt, and that the checkpoint hash matches, so a prompt can never be paired with the
wrong clip or the wrong detector.

Usage:
    python src/eval/protect_prompts_v2.py --checkpoint CKPT --out_dir results/anticlone/prompts/RQA \
        --dataset librispeech --data_root DATA SPLIT_ARGS --start 0 --stop 50
    python src/eval/protect_prompts_v2.py ... --merge
    python src/eval/protect_prompts_v2.py ... --diagnostic       (1 clip, verbose, writes nothing)
"""
import argparse
import hashlib
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

import clone_eval_v2 as ce          # noqa: E402  (dataset, crop, messages, detector, hashes: the evaluation's own code)
import cloner_watermark_eval as cwe  # noqa: E402
from quality_metrics import compute_pesq_stoi, compute_si_snr  # noqa: E402

SR = ce.SR
ARMS = ("wm", "rs", "q2")


def wav_sha(x):
    a = np.ascontiguousarray(x.detach().cpu().reshape(-1).numpy().astype(np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_flac(path, wav):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(wav.detach().cpu().reshape(-1).numpy().astype(np.float32), -1.0, 1.0)
    sf.write(path, x, SR, format="FLAC", subtype="PCM_16")


def read_audio(path, device):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device).reshape(1, 1, -1)


class Transcriber:
    """faster-whisper base.en on a float32 16 kHz array (same source and fallback as gen_whitebox_audio.py)."""
    def __init__(self, device, size="base.en"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(size, device=device, compute_type="float16" if device == "cuda" else "int8")

    def __call__(self, wav):
        x = np.ascontiguousarray(wav.detach().cpu().reshape(-1).numpy().astype(np.float32))
        segs, _ = self.m.transcribe(x, language="en", beam_size=5)
        t = " ".join(s.text for s in segs).strip()
        return t if t else "the quick brown fox jumps over the lazy dog"


def quality(clean, wav):
    c = clean.detach().cpu().reshape(-1).numpy().astype(np.float64)
    w = wav.detach().cpu().reshape(-1).numpy().astype(np.float64)
    pq, st = compute_pesq_stoi(c, w, SR)
    noise = w - c
    return {"pesq": float(pq), "stoi": float(st), "si_snr": float(compute_si_snr(c, w)),
            "snr_db": float(10 * np.log10(np.mean(c ** 2) / max(np.mean(noise ** 2), 1e-20))),
            "noise_linf": float(np.abs(noise).max())}


def merge(args, n_clips, settings):
    rec_dir = os.path.join(args.out_dir, "records")
    recs = []
    for i in range(n_clips):
        p = os.path.join(rec_dir, f"{i}.json")
        if os.path.exists(p):
            recs.append(json.load(open(p)))
    missing = sorted(set(range(n_clips)) - {r["index"] for r in recs})
    shas = {r["checkpoint_sha256"] for r in recs}
    assert len(shas) <= 1, f"records from different checkpoints in {args.out_dir}: {shas}"
    assert not shas or shas == {settings["checkpoint_sha256"]}, "--checkpoint differs from the one the prompts were made with"
    for a in ARMS:
        for r in recs:
            assert os.path.exists(os.path.join(args.out_dir, a, f"{r['index']}.flac")), (a, r["index"])
    summ = {}
    for a in ARMS:
        rs = [r["arms"][a] for r in recs]
        if rs:
            summ[a] = {k: float(np.mean([x[k] for x in rs])) for k in ("acc_source", "pesq", "stoi", "si_snr", "snr_db")}
            summ[a]["traced_source"] = float(np.mean([x["acc_source"] >= 13 / 16 for x in rs]))
    summ["q2_skipped_steps_total"] = int(sum(r["q2_diag"]["skipped"] for r in recs))
    summ["q2_clips_over_10_skipped"] = [r["index"] for r in recs if r["q2_diag"]["skipped"] > 10]
    out = {"script": "protect_prompts_v2", "settings": settings, "n_clips": n_clips, "n_done": len(recs),
           "missing": missing, "complete": not missing, "summary": summ,
           "records": sorted(recs, key=lambda r: r["index"])}
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"n_done": len(recs), "missing": missing[:20], "summary": summ}, indent=1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Omit for pretrained VoiceMark.")
    p.add_argument("--out_dir", required=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    # Q2 as registered (pre-registration-hexternal-2026-10-23.md); not to be retuned
    p.add_argument("--epsilon", type=float, default=0.015)
    p.add_argument("--shape_alpha", type=float, default=1.0)
    p.add_argument("--shape_floor", type=float, default=0.003)
    p.add_argument("--n_steps", type=int, default=40)
    p.add_argument("--step_size", type=float, default=0.0015)
    p.add_argument("--dis_to_wm_ratio", type=float, default=1.732)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--f5_steps", type=int, default=6)
    p.add_argument("--f5_cfg", type=float, default=2.0)
    p.add_argument("--pgd_text", default=ce.FIXED_TEXT)
    p.add_argument("--noise_seed_base", type=int, default=5000)
    p.add_argument("--f5_seed_base", type=int, default=7000)
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--prompt_seconds", default="3")
    # clip set: identical options and defaults to clone_eval_v2.py
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
    # --- T1 (transfer boosters) and T2 (verifier count), 2026-10-29. Defaults reproduce the registered Q2 exactly.
    p.add_argument("--momentum", type=float, default=0.0, help="MI-FGSM decay; 0 = off (Q2)")
    p.add_argument("--div_prob", type=float, default=0.0, help="input-diversity probability per step; 0 = off (Q2)")
    p.add_argument("--div_shift_ms", type=float, default=0.0, help="max input-diversity time shift, ms")
    p.add_argument("--div_gain", type=float, default=0.0, help="input-diversity gain jitter, fraction")
    p.add_argument("--verifiers", default=None,
                   help="comma-separated in-loop verifier set, e.g. 'ecapa' or 'ecapa,wavlm,xvec,resnet,wavlmbase'. "
                        "Default = the registered three. Resemblyzer is never allowed here.")
    p.add_argument("--only_arm", default=None,
                   help="write just the protected arm under this name (the T1/T2 arms); wm and rs are not rewritten")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--diagnostic", action="store_true")
    args = p.parse_args()
    if not args.diagnostic and not args.out_dir:
        p.error("--out_dir is required")

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_sha = ce.sha256(args.checkpoint)
    settings = {k: v for k, v in vars(args).items() if k not in ("start", "stop", "merge", "diagnostic")}
    settings.update(checkpoint_sha256=ck_sha, git=ce.git_info(), packages=ce.package_versions(),
                    arms={"wm": "watermark only", "rs": "random sign noise at Q2's bound (Q2's own start, signed)",
                          "q2": "Q2 white-box PGD (wens), as registered"})

    ds = ce.load_dataset(args, crop=args.max_full_seconds)
    n_clips = min(args.n_utterances, len(ds))
    if args.merge:
        merge(args, n_clips, settings)
        return

    import f5_whitebox_pgd as wb
    from f5_tts.api import F5TTS

    backbone = cwe.build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    f5a = wb.F5Attacker(F5TTS(device=device), steps=args.f5_steps, cfg=args.f5_cfg)
    ecapa = wb.load_ecapa_differentiable(device)
    spec = tuple(x.strip() for x in args.verifiers.split(",") if x.strip()) if args.verifiers else None
    need = set(spec) if spec else set(wb.DEFAULT_VERIFIERS)
    verifiers = wb.make_verifiers(
        ecapa,
        wb.load_xvector(device) if {"xvec"} & need else None,
        wb.load_wavlm_sv_differentiable(device) if {"wavlm"} & need else None,
        spec=spec,
        resnet=wb.load_resnet(device) if "resnet" in need else None,
        wavlm_base=wb.load_wavlm_base_sv(device) if "wavlmbase" in need else None)
    print("in-loop verifiers:", [n for n, _f in verifiers], flush=True)
    assert len(verifiers) >= 1
    transcribe = Transcriber(device)

    lo, hi = (0, 1) if args.diagnostic else (args.start, min(args.stop if args.stop is not None else n_clips, n_clips))
    rec_dir = os.path.join(args.out_dir, "records") if args.out_dir else None
    t0, n_new = time.time(), 0
    for i in range(lo, hi):
        rp = os.path.join(rec_dir, f"{i}.json") if rec_dir else None
        want = (args.only_arm,) if args.only_arm else ARMS
        if rp and os.path.exists(rp) and all(os.path.exists(os.path.join(args.out_dir, a, f"{i}.flac")) for a in want):
            old = json.load(open(rp))
            assert old["checkpoint_sha256"] == ck_sha, f"{rp} was made with another checkpoint"
            continue
        it = ds[i]
        full = ce.trim_padding(it["waveform"].reshape(-1))
        clean = ce.make_prompt(full, args.prompt_seconds, args.max_full_seconds).to(device).reshape(1, 1, -1)
        msg = ce.message_bits(args.message_seed_base + i, device)
        true_bits = np.array(msg[0].tolist())
        ref_text = transcribe(clean)

        _t = time.time()
        recon_wm, q2, _delta, diag = wb.whitebox_pgd(
            backbone, f5a, clean, msg, ref_text, args.pgd_text, args.epsilon, args.step_size, args.n_steps, "wens",
            lambda_wm=args.lambda_wm, dis_to_wm_ratio=args.dis_to_wm_ratio, ecapa_model=ecapa,
            noise_seed=args.noise_seed_base + i, f5_seed=args.f5_seed_base + i, verbose=args.diagnostic,
            verifiers=verifiers, shape_alpha=args.shape_alpha, shape_floor=args.shape_floor,
            momentum=args.momentum, div_prob=args.div_prob,
            div_shift=int(round(args.div_shift_ms * ce.SR / 1000.0)), div_gain=args.div_gain)
        pgd_sec = time.time() - _t
        with torch.no_grad():
            bound = wb.envelope_bound(recon_wm, args.epsilon, args.shape_alpha, args.shape_floor)
            start = wb._random_start(recon_wm, bound, args.noise_seed_base + i)     # Q2's own start
            sgn = torch.where(start >= 0, torch.ones_like(start), -torch.ones_like(start))
            rs = torch.clamp(recon_wm + bound * sgn, -1.0, 1.0)
            at_bound = float(((q2 - recon_wm).abs() >= 0.99 * bound).float().mean())

        out = {"wm": recon_wm, "rs": rs, "q2": q2}
        arms_here = ARMS
        if args.only_arm:                                   # T1/T2: one protected arm, its own folder
            out = {args.only_arm: q2}
            arms_here = (args.only_arm,)
        rec = {"index": i, "utterance_id": ce.utterance_id(ds, i, it), "speaker": str(it["speaker_id"]),
               "clean_sha": wav_sha(clean), "n_samples": int(clean.shape[-1]), "ref_text_pgd": ref_text,
               "checkpoint_sha256": ck_sha, "pgd_seconds": round(pgd_sec, 1), "arms": {},
               "q2_diag": {"skipped": diag["skipped"], "dis_loss_first": diag["dis_loss"][0] if diag["dis_loss"] else None,
                           "dis_loss_last": diag["dis_loss"][-1] if diag["dis_loss"] else None,
                           "wm_loss_last": diag["wm_loss"][-1] if diag["wm_loss"] else None,
                           "mean_bound": diag["mean_bound"], "share_at_bound": at_bound}}
        if not args.diagnostic:
            write_flac(os.path.join(args.out_dir, "clean", f"{i}.flac"), clean)   # reference for the pairing check
        for a in arms_here:
            if args.diagnostic:
                import tempfile
                path = os.path.join(tempfile.mkdtemp(), f"{a}.flac")
            else:
                path = os.path.join(args.out_dir, a, f"{i}.flac")
            write_flac(path, out[a])
            back = read_audio(path, device)                       # everything is measured on the file as an attacker gets it
            assert back.shape[-1] == clean.shape[-1], (a, back.shape, clean.shape)
            dec = np.array(ce.decode_bits(backbone, back))
            rec["arms"][a] = {"acc_source": float((dec == true_bits).mean()), **quality(clean, back)}
        if args.diagnostic:
            print(json.dumps(rec, indent=1))
            print(f"PGD time {pgd_sec:.1f} s for {args.n_steps} steps | peak GPU memory "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else "")
            if "wm" in rec["arms"]:
                assert rec["arms"]["wm"]["acc_source"] >= 0.85, "watermark not readable on the wm prompt: wrong checkpoint/rank?"
            assert diag["gn_dis"] and diag["gn_dis"][0] > 0, "zero disruption gradient: the F5 path is broken"
            if not diag["dis_loss"][-1] < diag["dis_loss"][0]:
                print("WARNING: PGD did not lower the in-loop similarity on this clip (one clip; look at the step log above)")
            print("DIAGNOSTIC OK")
            return
        os.makedirs(rec_dir, exist_ok=True)
        tmp = rp + ".tmp"
        json.dump(rec, open(tmp, "w"), indent=1)
        os.replace(tmp, rp)
        n_new += 1
        el = time.time() - t0
        a = rec["arms"]
        if args.only_arm:
            print(f"[{i}] {args.only_arm}: acc {a[args.only_arm]['acc_source']:.3f} | {pgd_sec:.0f} s", flush=True)
            continue
        print(f"[{i}] acc wm/rs/q2 {a['wm']['acc_source']:.3f}/{a['rs']['acc_source']:.3f}/{a['q2']['acc_source']:.3f} "
              f"pesq {a['wm']['pesq']:.2f}/{a['rs']['pesq']:.2f}/{a['q2']['pesq']:.2f} "
              f"dis {rec['q2_diag']['dis_loss_first']:.3f}->{rec['q2_diag']['dis_loss_last']:.3f} skip={diag['skipped']} "
              f"| {el / n_new:.0f} s/clip, ~{el / n_new * (hi - i - 1) / 60:.0f} min left", flush=True)
    print(f"DONE clips {lo}..{hi - 1} -> {args.out_dir}")


if __name__ == "__main__":
    main()
