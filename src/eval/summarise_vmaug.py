"""
src/eval/summarise_vmaug.py

Reads every result of the 2026-10-24 retraining (notebooks 24 and 25) that is
present in results/, prints the tables and applies the decision rule of
docs/preregistration_quality_cost_2026-10-24.md. Safe to run at any time: missing
files are reported, never guessed. The verdict is labelled INTERIM until both
F5-TTS and MaskGCT results exist for the arms involved.

File conventions (written by the notebooks):
  results/ce_{dataset}_{cloner}_{tag}_{protocol}.json   clone_eval_v2.py output
      dataset  libri | vctk
      protocol p3   (3 s prompt, fixed + unseen text)
               pdur (6 s and full-utterance prompts, fixed + unseen text)
  results/results_quality_paired_route2_vmaug_v3.json   quality_paired_ckpts.py, control R0
  results/results_quality_paired_stage1_vmaug.json      quality_paired_ckpts.py, Stage 1

Usage: python src/eval/summarise_vmaug.py [--results_dir results] [--out results/summary_vmaug_v3.json]
"""

import argparse
import glob
import json
import os
import re

import numpy as np

TAGS = ["pretrained", "stage1old_e29", "stage1legacy_e29", "stage1vmaug_e29", "stage1vmaugs1_e29",
        "route2old_r8", "R0", "R0s1", "R0f", "RQlo", "RQ", "RQhi", "RQA"]
DATASETS = ("libri", "ltc", "vctk")   # ours (train-clean-100 held-out), LibriSpeech test-clean, VCTK
PAPER_ACC = {"cosyvoice": 0.964, "f5tts": 0.979, "maskgct": 0.957}   # VoiceMark Table 1 (watermark ACC after cloning)
ARM_TAG = {a: f"route2vmaug_{a}" for a in ("R0", "R0s1", "R0f", "RQlo", "RQ", "RQhi", "RQA")}
CLONERS = ["yourtts", "xtts", "f5tts", "maskgct", "cosyvoice"]
REGISTERED = ["RQ", "RQA"]          # Holm over these two for the PESQ test
SWEEP = ["RQlo", "RQhi"]            # descriptive only
DECISIVE_CANDIDATES = ["xtts", "f5tts", "maskgct", "cosyvoice"]  # YourTTS is the training cloner: never decisive


def tag_of(t):
    return ARM_TAG.get(t, t)


def load_clone_results(rdir):
    out = {}
    for f in glob.glob(os.path.join(rdir, "ce_*.json")):
        m = re.match(r"ce_(libri|ltc|vctk)_([a-z0-9]+)_(.+)_(p3|pdur)\.json$", os.path.basename(f))
        if not m:
            continue
        try:
            out[m.groups()] = json.load(open(f))
        except Exception as e:
            print(f"[warn] unreadable {f}: {e}")
    return out


def get(ce, ds, cloner, tag, proto, key):
    r = ce.get((ds, cloner, tag_of(tag), proto))
    if not r:
        return None
    return r.get("summary", {}).get(key)


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def paired_clone_diff(ce, ds, cloner, tag_a, tag_b, key="3|unseen"):
    """Clip-paired ACC difference b - a (by clip index), descriptive."""
    from scipy import stats
    ra, rb = ce.get((ds, cloner, tag_of(tag_a), "p3")), ce.get((ds, cloner, tag_of(tag_b), "p3"))
    if not ra or not rb:
        return None
    p, t = key.split("|")
    da = {r["index"]: r["acc_clone"] for r in ra["records"] if r["status"] == "ok" and r["prompt"] == p and r["text_mode"] == t}
    db = {r["index"]: r["acc_clone"] for r in rb["records"] if r["status"] == "ok" and r["prompt"] == p and r["text_mode"] == t}
    idx = sorted(set(da) & set(db))
    if len(idx) < 5:
        return None
    a, b = np.array([da[i] for i in idx]), np.array([db[i] for i in idx])
    d = b - a
    out = {"n": len(idx), "diff": float(d.mean())}
    if np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(b, a).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/summary_vmaug_v3.json")
    args = ap.parse_args()
    from scipy import stats

    ce = load_clone_results(args.results_dir)
    summary = {"clone_tables": {}, "prompt_length": {}, "quality": {}, "verdicts": {}, "missing": []}
    print(f"found {len(ce)} clone result files")

    # ---------------------------------------------------------------- survival tables
    for ds in DATASETS:
        for cl in CLONERS:
            if not any(k[0] == ds and k[1] == cl for k in ce):
                continue
            print(f"\n=== survival: {ds} / {cl} (3 s prompt) ===")
            print(f"{'checkpoint':16s} | {'unseen: ACC  traced  FAR   SIM   WER  gate':44s} | {'fixed: ACC  traced':20s}")
            tab = {}
            for t in TAGS:
                u = get(ce, ds, cl, t, "p3", "3|unseen")
                f = get(ce, ds, cl, t, "p3", "3|fixed")
                if not u and not f:
                    continue
                u, f = u or {}, f or {}
                tab[t] = {"unseen": u, "fixed": f}
                print(f"{t:16s} | {fmt(u.get('acc_mean'))}  {fmt(u.get('traced_rate'), 2)}  {fmt(u.get('far_rate'), 2)}  "
                      f"{fmt(u.get('sim_ecapa_mean'))}  {fmt(u.get('wer_mean'), 2)}  {fmt(u.get('gate_pass')):4s} "
                      f"(n={u.get('n_ok', 0)}) | {fmt(f.get('acc_mean'))}  {fmt(f.get('traced_rate'), 2)}")
            summary["clone_tables"][f"{ds}/{cl}"] = tab

    # ---------------------------------------------------------------- Experiment A
    # Pretrained VoiceMark, one factor changed at a time from the baseline
    # (our clips, 3 s prompt, fixed sentence); see docs/preregistration_voicemark_audit_2026-10-24.md
    grid = [("baseline: ours, 3 s, fixed", "libri", "p3", "3|fixed"),
            ("text -> unseen", "libri", "p3", "3|unseen"),
            ("prompt -> 6 s", "libri", "pdur", "6|fixed"),
            ("prompt -> full utterance", "libri", "pdur", "full|fixed"),
            ("test set -> LibriSpeech test-clean", "ltc", "p3", "3|fixed"),
            ("test set -> VCTK", "vctk", "p3", "3|fixed"),
            ("paper-like: test-clean, full, unseen", "ltc", "pdur", "full|unseen")]
    summary["experiment_A"] = {}
    for cl in CLONERS:
        rows = [(name, get(ce, ds, cl, "pretrained", proto, key)) for name, ds, proto, key in grid]
        if not any(r for _, r in rows):
            continue
        base = rows[0][1]
        print(f"\n=== Experiment A: pretrained VoiceMark through {cl} (paper ACC: {PAPER_ACC.get(cl, '-')}) ===")
        print(f"{'condition':38s} {'ACC':>6s} {'d vs base':>9s} {'traced':>7s} {'FAR':>6s} {'SIM':>6s} {'WER':>6s} {'n':>4s}")
        out = {}
        for name, r in rows:
            if not r:
                print(f"{name:38s}  not run yet"); continue
            d = (r["acc_mean"] - base["acc_mean"]) if base and name != rows[0][0] else None
            print(f"{name:38s} {fmt(r.get('acc_mean')):>6s} {fmt(d):>9s} {fmt(r.get('traced_rate'), 2):>7s} "
                  f"{fmt(r.get('far_rate'), 2):>6s} {fmt(r.get('sim_ecapa_mean')):>6s} {fmt(r.get('wer_mean'), 2):>6s} "
                  f"{r.get('n_ok', 0):>4d}")
            out[name] = {"acc": r.get("acc_mean"), "d_vs_baseline": d, "traced": r.get("traced_rate"),
                         "far": r.get("far_rate"), "acc_ci_speaker": r.get("acc_ci_speaker"), "n": r.get("n_ok")}
        summary["experiment_A"][cl] = out

    # ---------------------------------------------------------------- Experiment B
    # Stage 1 trained three ways on the same data, epochs and clips: legacy (old code path),
    # corrected, corrected repeat (training noise); plus pretrained and the old unrecorded checkpoint.
    summary["experiment_B"] = {}
    for cl in CLONERS:
        rows = {t: get(ce, "libri", cl, t, "p3", "3|unseen") for t in
                ("pretrained", "stage1old_e29", "stage1legacy_e29", "stage1vmaug_e29", "stage1vmaugs1_e29")}
        if not rows.get("stage1legacy_e29") or not rows.get("stage1vmaug_e29"):
            continue
        fixed, rep_, leg = rows["stage1vmaug_e29"], rows.get("stage1vmaugs1_e29"), rows["stage1legacy_e29"]
        noise = abs(fixed["acc_mean"] - rep_["acc_mean"]) if rep_ else None
        tol = max(0.02, noise) if noise is not None else 0.02
        pd = paired_clone_diff(ce, "libri", cl, "stage1legacy_e29", "stage1vmaug_e29")
        diff = fixed["acc_mean"] - leg["acc_mean"]
        if pd and pd.get("wilcoxon_p", 1) < 0.05 and abs(diff) > tol:
            word = "improves" if diff > 0 else "worsens"
        else:
            word = "no detectable difference"
        print(f"\n=== Experiment B: Stage 1 training paths through {cl} (3 s, unseen text) ===")
        for t, r in rows.items():
            print(f"{t:20s} " + (f"ACC {fmt(r['acc_mean'])}  traced {fmt(r['traced_rate'], 2)}  FAR {fmt(r['far_rate'], 2)}"
                                 if r else "not run yet"))
        print(f"corrected - legacy: {diff:+.3f} (tolerance {tol:.3f}, training noise {fmt(noise)}, "
              f"paired {pd}) -> {word}")
        summary["experiment_B"][cl] = {"diff_corrected_minus_legacy": diff, "tolerance": tol,
                                       "training_noise": noise, "paired": pd, "reading": word}

    # ---------------------------------------------------------------- prompt length
    for ds in DATASETS:
        for cl in CLONERS:
            rows = {}
            for t in TAGS:
                vals = {}
                for spec, proto in (("3", "p3"), ("6", "pdur"), ("full", "pdur")):
                    for tm in ("unseen", "fixed"):
                        s = get(ce, ds, cl, t, proto, f"{spec}|{tm}")
                        if s:
                            vals[f"{spec}|{tm}"] = {"acc": s.get("acc_mean"), "traced": s.get("traced_rate"),
                                                    "prompt_s": s.get("prompt_seconds_mean")}
                if any(k.startswith("6|") or k.startswith("full|") for k in vals):
                    rows[t] = vals
            if rows:
                print(f"\n=== prompt length: {ds} / {cl} (ACC, traced; unseen text) ===")
                for t, v in rows.items():
                    print(f"{t:16s} " + "  ".join(f"{sp}: {fmt(v.get(sp + '|unseen', {}).get('acc'))}/"
                                                  f"{fmt(v.get(sp + '|unseen', {}).get('traced'), 2)}"
                                                  for sp in ("3", "6", "full")))
                summary["prompt_length"][f"{ds}/{cl}"] = rows

    # ---------------------------------------------------------------- quality
    qpath = os.path.join(args.results_dir, "results_quality_paired_route2_vmaug_v3.json")
    q = json.load(open(qpath)) if os.path.exists(qpath) else None
    if q:
        print("\n=== quality (LibriSpeech, 100 clips, watermarked vs clean) ===")
        cols = ["pesq", "stoi", "si_snr", "acc_clean", "acc_mask", "acc_shuffle", "acc_replace", "acc_encodec", "acc_perturb"]
        print(f"{'checkpoint':16s} " + " ".join(f"{c:>11s}" for c in cols))
        for name, s in q["summary"].items():
            print(f"{name:16s} " + " ".join(f"{fmt(s.get(c)):>11s}" for c in cols))
        summary["quality"] = q["summary"]
    else:
        summary["missing"].append(qpath)

    # ---------------------------------------------------------------- verdicts
    g1 = None
    if q and "R0" in q.get("summary", {}):
        g1 = q["summary"]["R0"].get("acc_clean", 0) >= 0.95
        print(f"\ngate G1 (R0 clean ACC >= 0.95): {fmt(q['summary']['R0'].get('acc_clean'))} -> {'pass' if g1 else 'FAIL'}")
        summary["gate_G1"] = g1
    if q and "R0" in q["per_clip"] and g1 is False:
        print("[verdict] not given: gate G1 failed (R0 lost clean-audio detection); the run is reported as failed")
    elif q and "R0" in q["per_clip"]:
        base = np.array(q["per_clip"]["R0"]["pesq"], float)
        pvals, dp = {}, {}
        for arm in REGISTERED + SWEEP:
            if arm not in q["per_clip"]:
                continue
            x = np.array(q["per_clip"][arm]["pesq"], float)
            ok = ~(np.isnan(base) | np.isnan(x))
            dp[arm] = float((x[ok] - base[ok]).mean())
            pvals[arm] = float(stats.wilcoxon(x[ok], base[ok], alternative="greater").pvalue) if np.any(x[ok] != base[ok]) else 1.0
        # Holm over the two registered arms
        reg = sorted([a for a in REGISTERED if a in pvals], key=lambda a: pvals[a])
        holm, running = {}, 0.0
        for k, a in enumerate(reg):
            running = max(running, min(1.0, pvals[a] * (len(reg) - k)))
            holm[a] = running

        decisive = [cl for cl in DECISIVE_CANDIDATES
                    if (get(ce, "libri", cl, "R0", "p3", "3|unseen") or {}).get("gate_pass") is True]
        final = all(get(ce, "libri", cl, "R0", "p3", "3|unseen") for cl in ("f5tts", "maskgct"))
        interim_cloners = decisive or (["xtts"] if get(ce, "libri", "xtts", "R0", "p3", "3|unseen") else [])
        label = "FINAL" if final and decisive else "INTERIM"
        print(f"\n=== pre-registered verdict ({label}) ===")
        print(f"decisive cloners (R0 passes the gate): {decisive or 'none yet'}"
              + ("" if decisive else f"; interim reading on {interim_cloners}"))

        for arm in REGISTERED + SWEEP:
            if arm not in dp:
                continue
            gain = dp[arm] >= 0.10 and (holm.get(arm, pvals[arm]) < 0.05)
            per = {}
            for cl in interim_cloners:
                r0 = get(ce, "libri", cl, "R0", "p3", "3|unseen")
                r1 = get(ce, "libri", cl, "R0s1", "p3", "3|unseen")
                ra = get(ce, "libri", cl, arm, "p3", "3|unseen")
                s1 = get(ce, "libri", cl, "stage1vmaug_e29", "p3", "3|unseen")
                if not r0 or not ra:
                    per[cl] = {"status": "missing results"}
                    continue
                tau = max(0.02, abs(r1["acc_mean"] - r0["acc_mean"])) if r1 else 0.02
                ttol = max(0.05, abs(r1["traced_rate"] - r0["traced_rate"])) if r1 else 0.05
                d_acc = ra["acc_mean"] - r0["acc_mean"]
                d_tr = ra["traced_rate"] - r0["traced_rate"]
                G = (r0["acc_mean"] - s1["acc_mean"]) if s1 else None
                kept = d_acc >= -tau and d_tr >= -ttol
                per[cl] = {"dACC": d_acc, "tau": tau, "dTraced": d_tr, "traced_tol": ttol, "kept": kept,
                           "recovery_gain_G": G,
                           "retention": ((ra["acc_mean"] - s1["acc_mean"]) / G) if (s1 and G and G > tau) else None,
                           "no_recovery_gain_to_preserve": (G is not None and G <= tau),
                           "r0s1_available": r1 is not None,
                           "paired": paired_clone_diff(ce, "libri", cl, "R0", arm)}
            if arm in SWEEP:
                verdict = "descriptive (sweep arm)"
            elif not gain:
                verdict = "NO QUALITY GAIN"
            elif all(v.get("kept") for v in per.values() if "kept" in v) and per:
                verdict = "PRESERVES"
            else:
                lost = [c for c, v in per.items() if v.get("kept") is False]
                verdict = f"QUALITY AT A SURVIVAL COST ({', '.join(lost)})" if lost else "QUALITY GAIN, survival not yet measured"
            summary["verdicts"][arm] = {"label": label, "dPESQ": dp[arm], "p_one_sided": pvals[arm],
                                        "p_holm": holm.get(arm), "quality_gain": gain, "per_cloner": per,
                                        "verdict": verdict}
            print(f"\n{arm}: dPESQ {dp[arm]:+.3f} (p {pvals[arm]:.2g}"
                  + (f", Holm {holm[arm]:.2g}" if arm in holm else "") + f") -> quality gain: {gain}")
            for cl, v in per.items():
                if "kept" in v:
                    print(f"   {cl:9s} dACC {v['dACC']:+.3f} (tol {v['tau']:.3f})  dTraced {v['dTraced']:+.3f} "
                          f"(tol {v['traced_tol']:.3f})  kept={v['kept']}  G={fmt(v['recovery_gain_G'])}"
                          + ("  [no recovery gain to preserve]" if v["no_recovery_gain_to_preserve"] else ""))
                else:
                    print(f"   {cl:9s} {v['status']}")
            print(f"   VERDICT ({label}): {verdict}")
    else:
        print("\n[verdict] quality results with R0 as control are not available yet")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
