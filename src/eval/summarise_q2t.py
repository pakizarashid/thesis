"""
src/eval/summarise_q2t.py  (2026-10-26)

Every number and verdict of docs/preregistration_q2t_2026-10-26.md. RQA watermark, unseen text, 100 clips (20 held-out
speakers), cloners F5-TTS, MaskGCT, CosyVoice. Arms: wm, rs, q2 (notebook 26), t, q2t (notebook 28).
Only complete result files are used; pairs are matched by clip index and checked by utterance ID.

    python src/eval/summarise_q2t.py --results_dir results --out results/summary_q2t.json
"""
import argparse
import json
import os

import numpy as np

from summarise_anticlone import boot_ci, holm, load, mcnemar, wil

CLONERS = ["f5tts", "maskgct", "cosyvoice"]
ARMS = ["wm", "rs", "q2", "t", "q2t"]
UNUSABLE = 0.5


def recs(R, cl, arm, suffix=""):
    d = load(f"{R}/ce_ac_{cl}_RQA_{arm}{suffix}.json")
    if not d:
        return None
    return {r["index"]: r for r in d["records"] if r["status"] == "ok" and r["text_mode"] == "unseen"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def pair(a, b, key):
    ks = sorted(k for k in set(a) & set(b) if a[k].get(key) is not None and b[k].get(key) is not None)
    for k in ks:
        assert a[k]["utterance_id"] == b[k]["utterance_id"], k
    return ks, np.array([a[k][key] for k in ks], float), np.array([b[k][key] for k in ks], float), [a[k]["speaker"] for k in ks]


def diff(a, b, key, alternative):
    ks, va, vb, spk = pair(a, b, key)
    d = va - vb
    return {"n": len(ks), "a": float(va.mean()), "b": float(vb.mean()), "diff": float(d.mean()), "ci_speaker": boot_ci(d, spk),
            "p": wil(d, alternative), "alternative": alternative}


def accept(r, thr):
    return {k: (v.get("sim_resemblyzer") is not None and v["sim_resemblyzer"] >= thr) for k, v in r.items()
            if v.get("sim_resemblyzer") is not None}


def mc(a, b):
    """paired acceptance: p, n(a yes, b no), n(a no, b yes)."""
    ks = sorted(set(a) & set(b))
    p, n10, n01 = mcnemar(np.array([a[k] for k in ks]), np.array([b[k] for k in ks]))
    return {"n": len(ks), "a": float(np.mean([a[k] for k in ks])), "b": float(np.mean([b[k] for k in ks])),
            "p_exact": p, "a_only": n10, "b_only": n01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/summary_q2t.json")
    a = ap.parse_args()
    R = a.results_dir
    S = {"calibration": None, "cells": {}, "HT1_wer": {}, "HT2_accept": {}, "HT3_similarity": {}, "HT4_watermark": {},
         "HT5_attribution": {}, "accept_tests": {}, "accept_curve": {}, "heldout_asr": {}, "quality": {}, "prompt_asr": None}

    cal = json.load(open(f"{R}/sv_calibration_resemblyzer.json")) if os.path.exists(f"{R}/sv_calibration_resemblyzer.json") else None
    thr = cal["thresholds"] if cal else {"deantifake_0697": 0.697}
    S["calibration"] = {"thresholds": thr, "test": cal["test_at_thresholds"] if cal else None,
                        "dev_eer": cal["development"]["eer"] if cal else None}
    T = thr.get("eer_dev", 0.697)
    print("== calibration ==")
    if cal:
        for k, v in cal["test_at_thresholds"].items():
            print(f"{k:16s} thr {v['threshold']:.3f}  test FAR {v['far']:.3f} {v['far_ci']}  FRR {v['frr']:.3f} {v['frr_ci']}")
    else:
        print("no calibration file: acceptance uses the De-AntiFake threshold 0.697 (provisional)")

    D = {(cl, arm): recs(R, cl, arm) for cl in CLONERS for arm in ARMS}
    print("\n== cells (unseen) ==  accepted = share of clones with Resemblyzer >= primary threshold")
    for (cl, arm), r in D.items():
        if not r:
            continue
        v = list(r.values()); acc = accept(r, T)
        k = sum(acc.values()); n = len(acc)
        S["cells"][f"{cl}|{arm}"] = {"n": len(v), "acc": float(np.mean([x["acc_clone"] for x in v])),
                                     "traced": float(np.mean([x["traced"] for x in v])),
                                     "sim_resemblyzer": float(np.nanmean([np.nan if x.get("sim_resemblyzer") is None else x["sim_resemblyzer"] for x in v])),
                                     "wer": float(np.nanmean([np.nan if x.get("wer") is None else x["wer"] for x in v])),
                                     "unusable": float(np.mean([(x.get("wer") or 0) >= UNUSABLE for x in v])),
                                     "accepted": k / max(n, 1), "accepted_ci": wilson(k, n),
                                     "accepted_by_threshold": {kk: float(np.mean(list(accept(r, tv).values()))) for kk, tv in thr.items()}}
        c = S["cells"][f"{cl}|{arm}"]
        print(f"{cl:9s} {arm:4s} n={c['n']:3d} RES {c['sim_resemblyzer']:.3f} accepted {c['accepted']:.2f} "
              f"[{c['accepted_ci'][0]:.2f},{c['accepted_ci'][1]:.2f}] WER {c['wer']:.3f} unusable {c['unusable']:.2f} "
              f"ACC {c['acc']:.3f} traced {c['traced']:.2f}")
        S["accept_curve"][f"{cl}|{arm}"] = {f"{t:.2f}": float(np.mean(list(accept(r, t).values()))) for t in np.arange(0.50, 0.951, 0.01)}

    # HT1 primary: clone WER q2t > q2, Holm over the three cloners
    cls = [cl for cl in CLONERS if D[(cl, "q2t")] and D[(cl, "q2")]]
    for cl in cls:
        S["HT1_wer"][cl] = diff(D[(cl, "q2t")], D[(cl, "q2")], "wer", "greater")
        ua = {k: (v.get("wer") or 0) >= UNUSABLE for k, v in D[(cl, "q2t")].items()}
        ub = {k: (v.get("wer") or 0) >= UNUSABLE for k, v in D[(cl, "q2")].items()}
        S["HT1_wer"][cl]["unusable"] = mc(ua, ub)
    for cl, pa in zip(cls, holm([S["HT1_wer"][c]["p"] for c in cls])):
        x = S["HT1_wer"][cl]; x["p_holm"] = pa
        x["verdict"] = "RAISES WER" if pa < 0.05 and x["diff"] >= 0.05 else "NO WER GAIN SHOWN"
    print("\n== HT1 (primary): clone WER, q2t - q2 ==")
    for cl, x in S["HT1_wer"].items():
        print(f"{cl:9s} {x['a']:.3f} vs {x['b']:.3f} diff {x['diff']:+.3f} CI [{x['ci_speaker'][0]:+.3f},{x['ci_speaker'][1]:+.3f}] "
              f"Holm p {x['p_holm']:.1e} unusable {x['unusable']['a']:.2f} vs {x['unusable']['b']:.2f} -> {x['verdict']}")

    # HT2: acceptance at the primary threshold, q2t vs q2 (two-sided McNemar, Holm)
    for cl in cls:
        S["HT2_accept"][cl] = mc(accept(D[(cl, "q2t")], T), accept(D[(cl, "q2")], T))
    for cl, pa in zip(cls, holm([S["HT2_accept"][c]["p_exact"] for c in cls])):
        x = S["HT2_accept"][cl]; x["p_holm"] = pa
        x["verdict"] = ("FEWER ACCEPTED" if x["a"] < x["b"] else "MORE ACCEPTED") if pa < 0.05 else "NO DIFFERENCE"
    print(f"\n== HT2: accepted as the speaker at {T:.3f}, q2t vs q2 ==")
    for cl, x in S["HT2_accept"].items():
        print(f"{cl:9s} {x['a']:.2f} vs {x['b']:.2f} (only q2t {x['a_only']}, only q2 {x['b_only']}) Holm p {x['p_holm']:.2e} -> {x['verdict']}")

    # descriptive acceptance tests (the calibrated version of notebook 26's H1)
    for cl in CLONERS:
        for x_, y_ in (("q2", "wm"), ("q2", "rs"), ("q2t", "wm"), ("t", "wm"), ("rs", "wm")):
            if D[(cl, x_)] and D[(cl, y_)]:
                S["accept_tests"][f"{cl}|{x_}_vs_{y_}"] = mc(accept(D[(cl, x_)], T), accept(D[(cl, y_)], T))
    print("\n== acceptance, paired (descriptive) ==")
    for k, x in S["accept_tests"].items():
        print(f"{k:22s} {x['a']:.2f} vs {x['b']:.2f}  exact McNemar p {x['p_exact']:.1e}")

    # HT3: held-out similarity q2t vs q2 (two-sided)
    for cl in cls:
        S["HT3_similarity"][cl] = diff(D[(cl, "q2")], D[(cl, "q2t")], "sim_resemblyzer", "two-sided")   # > 0: q2t lower
    print("\n== HT3: Resemblyzer, q2 - q2t (> 0 means q2t is less similar) ==")
    for cl, x in S["HT3_similarity"].items():
        print(f"{cl:9s} {x['diff']:+.3f} CI [{x['ci_speaker'][0]:+.3f},{x['ci_speaker'][1]:+.3f}] p {x['p']:.1e}")

    # HT4: watermark under q2t
    for cl in CLONERS:
        c = S["cells"].get(f"{cl}|q2t")
        if c:
            S["HT4_watermark"][cl] = {"acc": c["acc"], "traced": c["traced"],
                                      "verdict": "SURVIVES" if c["traced"] >= 0.80 and c["acc"] >= 0.90 else "DOES NOT SURVIVE"}
    print("\n== HT4: watermark under q2t ==", {k: (round(v["traced"], 2), v["verdict"]) for k, v in S["HT4_watermark"].items()})

    # HT5: the transcription attack alone
    for cl in CLONERS:
        if D[(cl, "t")] and D[(cl, "wm")]:
            S["HT5_attribution"][cl] = {"wer_t_minus_wm": diff(D[(cl, "t")], D[(cl, "wm")], "wer", "greater"),
                                        "res_wm_minus_t": diff(D[(cl, "wm")], D[(cl, "t")], "sim_resemblyzer", "greater")}
    print("\n== HT5: transcription attack alone (t vs wm) ==")
    for cl, x in S["HT5_attribution"].items():
        print(f"{cl:9s} WER {x['wer_t_minus_wm']['diff']:+.3f} (p {x['wer_t_minus_wm']['p']:.1e})  RES drop {x['res_wm_minus_t']['diff']:+.3f}")

    # held-out prompt recogniser (MaskGCT, CosyVoice transcribed with large-v3-turbo)
    for cl in ("maskgct", "cosyvoice"):
        a_, b_ = recs(R, cl, "q2t", "_asrturbo"), recs(R, cl, "q2", "_asrturbo")
        if a_ and b_:
            S["heldout_asr"][cl] = {"wer": diff(a_, b_, "wer", "greater"),
                                    "accept": mc(accept(a_, T), accept(b_, T))}
    if S["heldout_asr"]:
        print("\n== held-out prompt ASR (large-v3-turbo): q2t - q2 ==")
        for cl, x in S["heldout_asr"].items():
            print(f"{cl:9s} WER {x['wer']['a']:.3f} vs {x['wer']['b']:.3f} diff {x['wer']['diff']:+.3f} p {x['wer']['p']:.1e} | "
                  f"accepted {x['accept']['a']:.2f} vs {x['accept']['b']:.2f}")

    # prompt quality (paired with q2 on the same clips)
    P = os.path.join(R, "anticlone", "prompts", "RQA")
    m = os.path.join(P, "manifest.json")
    if os.path.exists(m):
        base = {r["index"]: r["arms"]["q2"] for r in json.load(open(m))["records"]}
        for arm in ("t", "q2t"):
            f = os.path.join(P, f"{arm}_manifest.json")
            if os.path.exists(f):
                rr = {r["index"]: r[arm] for r in json.load(open(f))["records"]}
                ks = sorted(set(rr) & set(base))
                S["quality"][arm] = {k: float(np.mean([rr[i][k] for i in ks])) for k in ("pesq", "stoi", "si_snr", "acc_source")}
                S["quality"][arm]["pesq_minus_q2"] = float(np.mean([rr[i]["pesq"] - base[i]["pesq"] for i in ks]))
        print("\n== prompt quality ==", json.dumps(S["quality"]))
        pa = os.path.join(R, "anticlone", "prompt_asr_RQA.json")
        if os.path.exists(pa):
            S["prompt_asr"] = json.load(open(pa))["summary_mean_wer"]
            print("== prompt WER (vs turbo transcript of the clean prompt) ==", json.dumps(S["prompt_asr"]))

    json.dump(S, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
