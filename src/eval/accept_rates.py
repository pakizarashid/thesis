"""
src/eval/accept_rates.py  (2026-10-28)

"Share of clones still accepted as the target speaker", for BOTH verifiers, at each verifier's own calibrated
threshold. No cloning and no new audio: every results/ce_ac_*.json already stores sim_resemblyzer and sim_ecapa
per clip, so once results/sv_calibration_{resemblyzer,ecapa}.json exist the rates are a read-off.

Why both:
    resemblyzer   held out of every loss          -> the honest number, the one that is reported as evidence
    ecapa         one of the three verifiers IN the perturbation loss -> the in-loop diagnostic

The two similarity scales are NOT comparable (different embedding spaces, different training objectives). What is
comparable is the DECISION: the share of clones at or above that verifier's own threshold, calibrated with the
identical genuine/impostor protocol on the same 40 development speakers. The gap between the two accept rates is
the quantity of interest: how much a self-graded evaluation would have inflated the protection result.

Reported per (cloner, watermark, arm): accept rate at eer_dev and at far1_dev, speaker-bootstrap 95 % CIs, and
for the protected arms an exact McNemar test against the watermark-only arm on the SAME clips.

    python src/eval/accept_rates.py --out results/accept_rates.json
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

VERIFIERS = ("resemblyzer", "ecapa")
FIELD = {"resemblyzer": "sim_resemblyzer", "ecapa": "sim_ecapa"}
ARM_ORDER = ["wm", "rs", "a0", "q2", "t", "q2t", "q2c"]
CLONER_ORDER = ["f5tts", "maskgct", "cosyvoice", "xtts", "yourtts"]
PAT = re.compile(r"^ce_ac_([a-z0-9]+)_([A-Za-z0-9]+)_([a-z0-9]+)\.json$")


def thresholds(results_dir):
    out = {}
    for v in VERIFIERS:
        p = os.path.join(results_dir, f"sv_calibration_{v}.json")
        if not os.path.exists(p):
            print(f"[skip] {v}: {p} missing (run sv_threshold_calibration.py --verifier {v})")
            continue
        d = json.load(open(p))
        # older runs wrote "resemblyzer (clone_eval_v2.Resem)" in this field
        assert v in str(d.get("verifier", "")), f"{p} is a calibration of {d.get('verifier')!r}, not {v!r}"
        keys = [k for k in d["thresholds"] if k in d["test_at_thresholds"]]
        out[v] = dict({k: d["thresholds"][k] for k in keys}, dev_eer=d["development"]["eer"],
                      keys=keys,
                      test={k: {kk: d["test_at_thresholds"][k][kk] for kk in ("far", "frr")} for k in keys})
    return out


def records(path, text_mode, prompt):
    d = json.load(open(path))
    if not d.get("complete"):
        return None
    return [r for r in d["records"]
            if r.get("status") == "ok" and r.get("text_mode") == text_mode and str(r.get("prompt")) == str(prompt)]


def boot_rate(flags, speakers, B=10000, seed=20261028):
    """95 % CI for a rate, resampling SPEAKERS (five clips of one speaker are not independent evidence)."""
    by = defaultdict(list)
    for f, s in zip(flags, speakers):
        by[s].append(f)
    keys = sorted(by)
    arr = {k: np.array(by[k], float) for k in keys}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        out.append(float(np.mean(np.concatenate([arr[k] for k in pick]))))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def mcnemar(a, b):
    """exact two-sided McNemar on paired 0/1 decisions (a = reference arm, b = protected arm)."""
    n01 = int(sum(1 for x, y in zip(a, b) if x and not y))      # accepted before, rejected after
    n10 = int(sum(1 for x, y in zip(a, b) if y and not x))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))
    return {"n01": n01, "n10": n10, "p": float(p)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out", default="results/accept_rates.json")
    p.add_argument("--text_mode", default="unseen")
    p.add_argument("--prompt", default="3")
    p.add_argument("--reference_arm", default="wm")
    a = p.parse_args()

    thr = thresholds(a.results_dir)
    assert thr, "no calibration file found: run sv_threshold_calibration.py first"

    cells = {}
    for f in sorted(glob.glob(os.path.join(a.results_dir, "ce_ac_*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue                                            # _asrturbo and other sensitivity files
        cl, wm, arm = m.groups()
        rr = records(f, a.text_mode, a.prompt)
        if not rr:
            continue
        cells[(cl, wm, arm)] = rr

    res = {"script": "accept_rates", "settings": vars(a), "thresholds": thr, "cells": {}}
    for (cl, wm, arm), rr in cells.items():
        row = {"n": len(rr), "traced_rate": float(np.mean([bool(r.get("traced")) for r in rr]))}
        ref = cells.get((cl, wm, a.reference_arm))
        for v, t in thr.items():
            sims = [r.get(FIELD[v]) for r in rr]
            if any(s is None for s in sims):
                continue
            spk = [r["speaker"] for r in rr]
            row[v] = {"sim_mean": float(np.mean(sims))}
            for key in t["keys"]:
                flags = [bool(s >= t[key]) for s in sims]
                row[v][key] = {"threshold": t[key], "accept": float(np.mean(flags)),
                               "accept_ci": boot_rate(flags, spk)}
                if ref is not None and arm != a.reference_arm:
                    by_ref = {r["index"]: r for r in ref}
                    pair = [(by_ref[r["index"]], r) for r in rr if r["index"] in by_ref
                            and by_ref[r["index"]].get(FIELD[v]) is not None]
                    if pair:
                        fa = [x[0][FIELD[v]] >= t[key] for x in pair]
                        fb = [x[1][FIELD[v]] >= t[key] for x in pair]
                        row[v][key]["vs_" + a.reference_arm] = dict(mcnemar(fa, fb), n_paired=len(pair))
        res["cells"][f"{cl}|{wm}|{arm}"] = row

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    # ------------------------------------------------------------------ the table for the slide
    def order(k):
        cl, wm, arm = k.split("|")
        return (CLONER_ORDER.index(cl) if cl in CLONER_ORDER else 9, wm,
                ARM_ORDER.index(arm) if arm in ARM_ORDER else 9)

    for v in [x for x in VERIFIERS if x in thr]:
        t = thr[v]
        tag = "IN the perturbation loss - diagnostic only" if v == "ecapa" else "held out of every loss - the honest number"
        print(f"\n=== {v}  ({tag})")
        print(f"    threshold {t['eer_dev']:.4f} (dev EER {t['dev_eer']:.3f}); on the 20 test speakers "
              f"FAR {t['test']['eer_dev']['far']:.3f} FRR {t['test']['eer_dev']['frr']:.3f}")
        print(f"    {'cloner':10s} {'wm':10s} {'arm':5s} {'n':>4s} {'sim':>7s} {'accept':>8s} {'95% CI':>15s}   vs wm (McNemar)")
        for k in sorted(res["cells"], key=order):
            row = res["cells"][k]
            if v not in row:
                continue
            cl, wm, arm = k.split("|")
            e = row[v]["eer_dev"]
            mc = e.get("vs_wm")
            s = (f"    {cl:10s} {wm:10s} {arm:5s} {row['n']:4d} {row[v]['sim_mean']:7.4f} {e['accept']:8.3f} "
                 f"  [{e['accept_ci'][0]:.3f},{e['accept_ci'][1]:.3f}]")
            if mc:
                s += f"   p={mc['p']:.2e}  ({mc['n01']} lost / {mc['n10']} gained)"
            print(s)

    both = [v for v in VERIFIERS if v in thr]
    if len(both) == 2:
        print("\n=== how much the in-loop verifier inflates the result (accept rate at each verifier's own threshold)")
        print(f"    {'cloner':10s} {'wm':10s} {'arm':5s} {'held-out':>9s} {'in-loop':>8s} {'inflation':>10s}")
        for k in sorted(res["cells"], key=order):
            row = res["cells"][k]
            if not all(v in row for v in both):
                continue
            cl, wm, arm = k.split("|")
            h = row["resemblyzer"]["eer_dev"]["accept"]
            i = row["ecapa"]["eer_dev"]["accept"]
            print(f"    {cl:10s} {wm:10s} {arm:5s} {h:9.3f} {i:8.3f} {h - i:+10.3f}")
        print("    inflation = how many more clones the honest verifier still accepts than the self-graded one would admit")

    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
