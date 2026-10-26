"""
src/eval/summarise_anticlone.py  (anti-cloning composability, 2026-10-26)

Reads
    results/anticlone/prompts/{tag}/manifest.json         (protect_prompts_v2.py)
    results/ce_ac_{cloner}_{tag}_{arm}.json               (clone_eval_v2.py --prompt_dir ... --resemblyzer)
and applies the decision rules of docs/preregistration_anticlone_2026-10-26.md. Only files marked complete are used;
everything is paired by clip index (and checked by utterance ID).

    python src/eval/summarise_anticlone.py --results_dir results --out results/summary_anticlone.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon

TAGS = ["pretrained", "R0", "RQA"]
ARMS = ["wm", "rs", "q2"]
EXTRA_ARMS = ["a0", "q2c"]   # a0: YourTTS-surrogate perturbation (surrogate comparison); q2c: Q2 + intelligibility term. RQA only
CLONERS = ["f5tts", "maskgct", "cosyvoice", "xtts", "yourtts"]
N_BOOT, SEED = 10000, 20261026
SENS_FROM = 40          # clips 0-39 were Q2's development clips


def load(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return d if d.get("complete") else None


def boot_ci(diff, spk):
    diff, spk = np.asarray(diff, float), np.asarray(spk)
    groups = [diff[spk == s] for s in np.unique(spk)]
    rng = np.random.default_rng(SEED)
    m = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(N_BOOT)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def wil(x, alternative):
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    return float(wilcoxon(x, alternative=alternative, zero_method="wilcox").pvalue)


def mcnemar(a, b):
    """a, b: boolean arrays (traced). Exact two-sided McNemar on discordant pairs."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return float(binomtest(n10, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0, n10, n01


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, run = [None] * len(ps), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[k]))
        adj[k] = run
    return adj


def per_clip(d, mode):
    out = {}
    for r in d["records"]:
        if r["status"] == "ok" and r["text_mode"] == mode:
            out[r["index"]] = r
    return out


def paired(a, b, key, idx_min=0):
    """a, b: {index: record}. Returns (indices, va, vb, speakers) over clips present in both with the key."""
    ks = sorted(k for k in set(a) & set(b) if k >= idx_min and a[k].get(key) is not None and b[k].get(key) is not None)
    for k in ks:
        assert a[k]["utterance_id"] == b[k]["utterance_id"], k
    return ks, np.array([a[k][key] for k in ks], float), np.array([b[k][key] for k in ks], float), [a[k]["speaker"] for k in ks]


def drop(a, b, idx_min=0):
    """Held-out similarity drop a - b (b = the protected arm). One-sided: a > b."""
    ks, va, vb, spk = paired(a, b, "sim_resemblyzer", idx_min)
    if len(ks) < 10:
        return None
    d = va - vb
    return {"n": len(ks), "mean": float(d.mean()), "ci_speaker": boot_ci(d, spk), "p_one_sided": wil(d, "greater"),
            "positive_speakers": int(sum(np.mean(d[np.array(spk) == s]) > 0 for s in set(spk))), "n_speakers": len(set(spk))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/summary_anticlone.json")
    a = ap.parse_args()
    R = a.results_dir
    S = {"prompts": {}, "cells": {}, "gate": {}, "H1": {}, "H2": {}, "H3": {}, "H5": {}, "sensitivity_40_99": {}}

    # ---- protected prompts: quality and source ACC
    man = {}
    for t in TAGS:
        m = load(f"{R}/anticlone/prompts/{t}/manifest.json")
        if m:
            man[t] = {r["index"]: r for r in m["records"]}
            S["prompts"][t] = m["summary"]
            for xa in EXTRA_ARMS:
                xm = load(f"{R}/anticlone/prompts/{t}/{xa}_manifest.json")
                if xm:
                    S["prompts"][t][xa] = xm["summary"][xa]
                    for r in xm["records"]:
                        if r["index"] in man[t]:
                            man[t][r["index"]]["arms"][xa] = r[xa]
    print("\n== protected prompts (3 s, vs clean; on the saved 16-bit files) ==")
    print(f"{'tag':11s} {'arm':3s} {'PESQ':>6s} {'STOI':>6s} {'SI-SNR':>7s} {'SNR':>6s} {'srcACC':>7s}")
    for t in man:
        for arm in ARMS + EXTRA_ARMS:
            if arm not in S["prompts"][t]:
                continue
            s = S["prompts"][t][arm]
            print(f"{t:11s} {arm:3s} {s['pesq']:6.3f} {s['stoi']:6.3f} {s['si_snr']:7.2f} {s['snr_db']:6.2f} {s['acc_source']:7.3f}")
    if "RQA" in man and "pretrained" in man:
        for arm in ARMS:
            ks = sorted(set(man["RQA"]) & set(man["pretrained"]))
            d = np.array([man["RQA"][k]["arms"][arm]["pesq"] - man["pretrained"][k]["arms"][arm]["pesq"] for k in ks])
            S["prompts"].setdefault("pesq_RQA_minus_pretrained", {})[arm] = {"mean": float(d.mean()), "p_two_sided": wil(d, "two-sided")}
        print("PESQ RQA - official:", {k: round(v["mean"], 3) for k, v in S["prompts"]["pesq_RQA_minus_pretrained"].items()})

    # ---- clone results
    D = {}
    for cl in CLONERS:
        for t in TAGS:
            for arm in ARMS + EXTRA_ARMS:
                d = load(f"{R}/ce_ac_{cl}_{t}_{arm}.json")
                if d:
                    D[(cl, t, arm)] = d
    modes = ["unseen", "fixed"]
    print("\n== clones (3 s prompt): ACC / traced / SIM held-out (Resemblyzer) / SIM ECAPA (in Q2's loss) / WER ==")
    for (cl, t, arm), d in sorted(D.items()):
        for mode in modes:
            s = d["summary"].get(f"3|{mode}")
            if not s:
                continue
            S["cells"][f"{cl}|{t}|{arm}|{mode}"] = s
            print(f"{cl:9s} {t:10s} {arm:3s} {mode:6s} n={s['n_ok']:3d} ACC {s['acc_mean']:.3f} traced {s['traced_rate']:.2f} "
                  f"res {s.get('sim_resemblyzer_mean', float('nan')):.3f} ecapa {s.get('sim_ecapa_mean', float('nan')):.3f} "
                  f"WER {s.get('wer_mean', float('nan')):.3f}")

    # ---- gate (RQA, watermark only, unseen text) decides the decisive cloners
    decisive = []
    for cl in CLONERS:
        s = S["cells"].get(f"{cl}|RQA|wm|unseen")
        if s:
            ok = bool(s.get("sim_ecapa_mean", 0) > 0.30 and s.get("wer_mean", 1) < 0.20)
            S["gate"][cl] = {"sim_ecapa": s.get("sim_ecapa_mean"), "wer": s.get("wer_mean"), "pass": ok}
            if ok and cl in ("f5tts", "maskgct", "cosyvoice"):
                decisive.append(cl)
    S["decisive_cloners"] = decisive
    print("\ngate (RQA, watermark only, unseen):", S["gate"], "-> decisive:", decisive)

    def rec(cl, t, arm, mode="unseen"):
        d = D.get((cl, t, arm))
        return per_clip(d, mode) if d else None

    # ---- H1: protection (held-out similarity drop), every tag, every cloner; decisive = RQA on decisive cloners
    print("\n== H1: held-out similarity drop (unseen text) ==")
    for cl in CLONERS:
        for t in TAGS:
            for mode in modes:
                wm, rs, q2 = rec(cl, t, "wm", mode), rec(cl, t, "rs", mode), rec(cl, t, "q2", mode)
                if not (wm and q2):
                    continue
                h = {"wm_minus_q2": drop(wm, q2), "rs_minus_q2": drop(rs, q2) if rs else None,
                     "wm_minus_rs": drop(wm, rs) if rs else None}
                x = h["wm_minus_q2"]
                if x:
                    h["verdict"] = "PROTECTS" if (x["p_one_sided"] < 0.05 and x["ci_speaker"][0] > 0) else "NOT SHOWN"
                    y = h["rs_minus_q2"]
                    h["beyond_random_noise"] = bool(y and y["p_one_sided"] < 0.05 and y["ci_speaker"][0] > 0)
                S["H1"][f"{cl}|{t}|{mode}"] = h
                if x:
                    f = lambda z: (f"{z['mean']:+.4f} [{z['ci_speaker'][0]:+.4f},{z['ci_speaker'][1]:+.4f}] p {z['p_one_sided']:.1e}" if z else "-")
                    print(f"{cl:9s} {t:10s} {mode:6s} wm-q2 {f(x)} | rs-q2 {f(h['rs_minus_q2'])} | wm-rs {f(h['wm_minus_rs'])} -> {h['verdict']}"
                          + (" (beyond random noise)" if h["beyond_random_noise"] else ""))
                if mode == "unseen":
                    S["sensitivity_40_99"][f"{cl}|{t}"] = drop(wm, q2, SENS_FROM)

    # ---- surrogate comparison (pre-registration addendum): Q2 (F5-TTS surrogate) vs A0 (YourTTS surrogate), RQA
    S["surrogate"] = {}
    print("\n== surrogate comparison, RQA: held-out drop of A0 (YourTTS surrogate) and Q2 (F5-TTS surrogate) ==")
    for cl in CLONERS:
        for mode in modes:
            wm, a0, q2 = rec(cl, "RQA", "wm", mode), rec(cl, "RQA", "a0", mode), rec(cl, "RQA", "q2", mode)
            if not (wm and a0 and q2):
                continue
            h = {"wm_minus_a0": drop(wm, a0), "a0_minus_q2": drop(a0, q2)}
            ks, aa, aq, _ = paired(a0, q2, "acc_clone")
            h["acc_a0"], h["acc_q2"] = float(aa.mean()), float(aq.mean())
            h["traced_a0"] = float(np.mean([a0[k]["traced"] for k in ks]))
            h["traced_q2"] = float(np.mean([q2[k]["traced"] for k in ks]))
            x = h["a0_minus_q2"]
            if x:
                h["verdict"] = ("F5 SURROGATE TRANSFERS BETTER" if x["p_one_sided"] < 0.05 and x["ci_speaker"][0] > 0
                                else "NO DIFFERENCE SHOWN")
            S["surrogate"][f"{cl}|{mode}"] = h
            f = lambda z: (f"{z['mean']:+.4f} [{z['ci_speaker'][0]:+.4f},{z['ci_speaker'][1]:+.4f}] p {z['p_one_sided']:.1e}" if z else "-")
            print(f"{cl:9s} {mode:6s} wm-a0 {f(h['wm_minus_a0'])} | a0-q2 {f(x)} | ACC a0 {h['acc_a0']:.3f} q2 {h['acc_q2']:.3f} "
                  f"traced {h['traced_a0']:.2f}/{h['traced_q2']:.2f} -> {h.get('verdict')}")
    if "RQA" in man and all("a0" in man["RQA"][k]["arms"] for k in man["RQA"]):
        d = np.array([man["RQA"][k]["arms"]["q2"]["pesq"] - man["RQA"][k]["arms"]["a0"]["pesq"] for k in sorted(man["RQA"])])
        S["surrogate"]["pesq_q2_minus_a0"] = {"mean": float(d.mean()), "p_two_sided": wil(d, "two-sided")}
        print("PESQ q2 - a0 (RQA):", round(float(d.mean()), 3))

    # ---- intelligibility addendum: q2c (Q2 + wav2vec2 CTC term) vs q2, RQA. WER is scored by Whisper (not in the loss)
    S["content"] = {}
    print("\n== intelligibility addendum, RQA: q2c vs q2 (WER by Whisper; CTC of wav2vec2 was in the loss) ==")
    for cl in CLONERS:
        for mode in modes:
            q2, q2c, wm = rec(cl, "RQA", "q2", mode), rec(cl, "RQA", "q2c", mode), rec(cl, "RQA", "wm", mode)
            if not (q2 and q2c):
                continue
            ks, wq, wc, spk = paired(q2c, q2, "wer")
            d = wq - wc
            h = {"n": len(ks), "wer_q2c": float(wq.mean()), "wer_q2": float(wc.mean()), "wer_diff": float(d.mean()),
                 "wer_ci_speaker": boot_ci(d, spk), "wer_p_one_sided": wil(d, "greater"),
                 "unusable_q2c": float(np.mean(wq > 0.5)), "unusable_q2": float(np.mean(wc > 0.5)),
                 "res_drop_q2_minus_q2c": drop(q2, q2c), "wm_minus_q2c": drop(wm, q2c) if wm else None}
            ks2, a_c, a_q, _ = paired(q2c, q2, "acc_clone")
            h["acc_q2c"], h["acc_q2"] = float(a_c.mean()), float(a_q.mean())
            h["traced_q2c"] = float(np.mean([q2c[k]["traced"] for k in ks2]))
            h["verdict"] = ("RAISES WER" if h["wer_diff"] >= 0.10 and h["wer_p_one_sided"] < 0.05 and h["wer_ci_speaker"][0] > 0
                            else "NO WER GAIN SHOWN")
            S["content"][f"{cl}|{mode}"] = h
            print(f"{cl:9s} {mode:6s} WER q2c {h['wer_q2c']:.3f} vs q2 {h['wer_q2']:.3f} (+{h['wer_diff']:.3f}, p {h['wer_p_one_sided']:.1e}) "
                  f"WER>0.5 {h['unusable_q2c']:.2f}/{h['unusable_q2']:.2f} ACC {h['acc_q2c']:.3f}/{h['acc_q2']:.3f} "
                  f"traced q2c {h['traced_q2c']:.2f} -> {h['verdict']}")
    if "RQA" in man and all("q2c" in man["RQA"][k]["arms"] for k in man["RQA"]):
        d = np.array([man["RQA"][k]["arms"]["q2c"]["pesq"] - man["RQA"][k]["arms"]["q2"]["pesq"] for k in sorted(man["RQA"])])
        S["content"]["pesq_q2c_minus_q2"] = {"mean": float(d.mean()), "p_two_sided": wil(d, "two-sided")}
        print("PESQ q2c - q2 (RQA):", round(float(d.mean()), 3))

    # ---- H2: the watermark survives protection (RQA, decisive cloners)
    print("\n== H2: watermark under protection (RQA, unseen) ==")
    for cl in CLONERS:
        wm, q2 = rec(cl, "RQA", "wm"), rec(cl, "RQA", "q2")
        if not (wm and q2):
            continue
        ks, aw, aq, _ = paired(wm, q2, "acc_clone")
        tw = np.array([wm[k]["traced"] for k in ks]); tq = np.array([q2[k]["traced"] for k in ks])
        p_tr, n10, n01 = mcnemar(tw, tq)
        h = {"acc_q2": float(aq.mean()), "traced_q2": float(tq.mean()), "acc_drop_vs_wm": float((aw - aq).mean()),
             "acc_drop_p_two_sided": wil(aw - aq, "two-sided"), "traced_drop_vs_wm": float(tw.mean() - tq.mean()),
             "traced_mcnemar_p": p_tr, "lost_traced": n10, "gained_traced": n01}
        h["verdict"] = "SURVIVES" if (h["traced_q2"] >= 0.80 and h["acc_q2"] >= 0.90) else "DOES NOT SURVIVE"
        h["decisive"] = cl in decisive
        S["H2"][cl] = h
        print(cl, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()})

    # ---- H3: clone-aware vs official VoiceMark under the same protection (q2 arm), Holm over decisive cloners
    print("\n== H3: clone-aware vs official VoiceMark, both protected with Q2 (unseen) ==")
    for t in ["RQA", "R0"]:
        rows, ps = [], []
        for cl in CLONERS:
            a_, b_ = rec(cl, t, "q2"), rec(cl, "pretrained", "q2")
            if not (a_ and b_):
                continue
            ks, va, vb, spk = paired(a_, b_, "acc_clone")
            ta = np.array([a_[k]["traced"] for k in ks]); tb = np.array([b_[k]["traced"] for k in ks])
            p_tr, _, _ = mcnemar(tb, ta)
            r = {"cloner": cl, "acc_tag": float(va.mean()), "acc_official": float(vb.mean()), "diff": float((va - vb).mean()),
                 "ci_speaker": boot_ci(va - vb, spk), "p_two_sided": wil(va - vb, "two-sided"),
                 "traced_tag": float(ta.mean()), "traced_official": float(tb.mean()), "traced_mcnemar_p": p_tr,
                 "decisive": cl in decisive}
            rows.append(r)
            if r["decisive"]:
                ps.append(r["p_two_sided"])
        adj = iter(holm(ps)) if ps else iter([])
        for r in rows:
            if r["decisive"]:
                r["p_holm"] = next(adj)
                r["verdict"] = ("BETTER" if r["diff"] > 0 else "WORSE") if r["p_holm"] < 0.05 else "NO DIFFERENCE"
            print(t, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        S["H3"][t] = rows

    # ---- H5 (descriptive): is protection weaker on the clone-aware watermark?
    print("\n== H5 (descriptive): drop(RQA) - drop(official), unseen ==")
    for cl in CLONERS:
        w1, q1, w0, q0 = rec(cl, "RQA", "wm"), rec(cl, "RQA", "q2"), rec(cl, "pretrained", "wm"), rec(cl, "pretrained", "q2")
        if not all([w1, q1, w0, q0]):
            continue
        ks = sorted(set(w1) & set(q1) & set(w0) & set(q0))
        ks = [k for k in ks if all(x[k].get("sim_resemblyzer") is not None for x in (w1, q1, w0, q0))]
        d = np.array([(w1[k]["sim_resemblyzer"] - q1[k]["sim_resemblyzer"]) - (w0[k]["sim_resemblyzer"] - q0[k]["sim_resemblyzer"]) for k in ks])
        S["H5"][cl] = {"n": len(ks), "mean": float(d.mean()), "ci_speaker": boot_ci(d, [w1[k]["speaker"] for k in ks]),
                       "p_two_sided": wil(d, "two-sided")}
        print(cl, S["H5"][cl])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(S, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
