"""
src/eval/summarise_f5loop.py  (2026-10-26)

Decision rules of docs/preregistration_f5loop_2026-10-26.md: R0 trained with F5-TTS in the loop (R0F5, seed 0;
R0F5s1, seed 1) against R0 trained with YourTTS in the loop (R0, R0s1) and official VoiceMark, on the same
100 clips, 3 s prompt, unseen text (fixed text reported too). Only complete files are used; pairs are checked
by utterance ID.

    python src/eval/summarise_f5loop.py --results_dir results --out results/summary_f5loop.json
"""
import argparse
import json
import os

import numpy as np

from summarise_anticlone import boot_ci, holm, load, mcnemar, wil

CLONERS = ["f5tts", "maskgct", "cosyvoice", "xtts", "yourtts"]
WEAK, STRONG = ["xtts", "yourtts"], ["f5tts", "maskgct", "cosyvoice"]
TAG = {"official": "pretrained", "R0": "route2vmaug_R0", "R0s1": "route2vmaug_R0s1",
       "R0F5": "route2vmaug_R0F5", "R0F5s1": "route2vmaug_R0F5s1"}


def recs(R, cl, arm, mode):
    d = load(f"{R}/ce_libri_{cl}_{TAG[arm]}_p3.json")
    if not d:
        return None
    return {r["index"]: r for r in d["records"] if r["status"] == "ok" and r["text_mode"] == mode and r["prompt"] == "3"}


def compare(a, b):
    ks = sorted(set(a) & set(b))
    for k in ks:
        assert a[k]["utterance_id"] == b[k]["utterance_id"], k
    va = np.array([a[k]["acc_clone"] for k in ks]); vb = np.array([b[k]["acc_clone"] for k in ks])
    ta = np.array([a[k]["traced"] for k in ks]); tb = np.array([b[k]["traced"] for k in ks])
    spk = [a[k]["speaker"] for k in ks]
    p_tr, _, _ = mcnemar(tb, ta)
    return {"n": len(ks), "acc_a": float(va.mean()), "acc_b": float(vb.mean()), "diff": float((va - vb).mean()),
            "ci_speaker": boot_ci(va - vb, spk), "p_two_sided": wil(va - vb, "two-sided"),
            "traced_a": float(ta.mean()), "traced_b": float(tb.mean()), "traced_mcnemar_p": p_tr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/summary_f5loop.json")
    a = ap.parse_args()
    R, S = a.results_dir, {"cells": {}, "R0F5_vs_R0": {}, "R0F5_vs_official": {}, "seed_noise": {}, "verdict": {}, "quality": {}}

    for mode in ("unseen", "fixed"):
        for cl in CLONERS:
            for arm in TAG:
                r = recs(R, cl, arm, mode)
                if r:
                    v = [x["acc_clone"] for x in r.values()]
                    S["cells"][f"{cl}|{arm}|{mode}"] = {"n": len(v), "acc": float(np.mean(v)),
                                                        "traced": float(np.mean([x["traced"] for x in r.values()])),
                                                        "sim_ecapa": float(np.mean([x.get("sim_ecapa", np.nan) for x in r.values()])),
                                                        "wer": float(np.mean([x.get("wer", np.nan) for x in r.values()]))}
    print("== ACC / traced (3 s prompt) ==")
    for k, v in S["cells"].items():
        print(f"{k:32s} n={v['n']:3d} ACC {v['acc']:.3f} traced {v['traced']:.2f} SIM {v['sim_ecapa']:.3f} WER {v['wer']:.3f}")

    for cl in CLONERS:
        r0, r0s1, f5, f5s1, off = (recs(R, cl, x, "unseen") for x in ("R0", "R0s1", "R0F5", "R0F5s1", "official"))
        if f5 and r0:
            S["R0F5_vs_R0"][cl] = compare(f5, r0)
        if f5 and off:
            S["R0F5_vs_official"][cl] = compare(f5, off)
        noise = []
        if r0 and r0s1:
            noise.append(abs(compare(r0, r0s1)["diff"]))
        if f5 and f5s1:
            noise.append(abs(compare(f5, f5s1)["diff"]))
        S["seed_noise"][cl] = {"values": noise, "tol": max([0.02] + noise)}

    print("\n== R0F5 - R0 (unseen), Holm within each group ==")
    for group, name in ((WEAK, "weak"), (STRONG, "strong")):
        cls = [c for c in group if c in S["R0F5_vs_R0"]]
        adj = holm([S["R0F5_vs_R0"][c]["p_two_sided"] for c in cls]) if cls else []
        for c, pa in zip(cls, adj):
            x = S["R0F5_vs_R0"][c]; tol = S["seed_noise"][c]["tol"]
            x["p_holm"], x["tol"] = pa, tol
            if pa < 0.05 and x["diff"] > tol:
                v = "F5 IN THE LOOP IS BETTER"
            elif pa < 0.05 and x["diff"] < -tol:
                v = "F5 IN THE LOOP IS WORSE"
            else:
                v = "NO DIFFERENCE"
            S["verdict"][c] = v
            print(f"{name:6s} {c:9s} R0F5 {x['acc_a']:.3f} vs R0 {x['acc_b']:.3f} diff {x['diff']:+.3f} "
                  f"CI [{x['ci_speaker'][0]:+.3f},{x['ci_speaker'][1]:+.3f}] Holm p {pa:.1e} tol {tol:.3f} -> {v}")
    print("\n== R0F5 - official (unseen) ==")
    for c, x in S["R0F5_vs_official"].items():
        print(f"{c:9s} diff {x['diff']:+.3f} p {x['p_two_sided']:.1e} traced {x['traced_a']:.2f} vs {x['traced_b']:.2f}")

    q = f"{R}/results_quality_paired_f5loop.json"
    if os.path.exists(q):
        S["quality"] = json.load(open(q)).get("summary", json.load(open(q)))
        print("\nquality file:", q)
    json.dump(S, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
