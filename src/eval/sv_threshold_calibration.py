"""
src/eval/sv_threshold_calibration.py  (2026-10-26, pre-registered in docs/preregistration_q2t_2026-10-26.md, part A)

A decision threshold for the held-out verifier (Resemblyzer), so that "the clone is accepted as the target speaker"
can be counted instead of only reporting a similarity drop.

Scoring is the evaluation's own: clone_eval_v2.Resem (same VoiceEncoder, same preprocess_wav, same cosine), the
enrolment is the whole trimmed utterance (clone_eval_v2: trim_padding of the item, crop 20 s), exactly what a clone
is compared with. The probe is a DIFFERENT real recording, cut to a clone-like length:

--verifier ecapa runs the IDENTICAL protocol on clone_eval_v2.Ecapa (the same speechbrain encoder, the same
encode_batch on the same trimmed utterance, the same cosine), so its threshold is on the scale of the sim_ecapa
values already stored in every results/ce_ac_*.json. ECAPA is one of the three verifiers INSIDE the perturbation
loss, so its accept rate is reported only as the in-loop diagnostic that quantifies how much a self-graded
evaluation inflates a protection result; the honest number stays Resemblyzer's, which is in no loss.

    genuine pair  : enrolment = utterance u of speaker s, probe = another utterance v of s
    impostor pair : enrolment = utterance u of speaker s, probe = an utterance of another speaker
    probe length  : drawn from the clone lengths of the three strong cloners (RQA, watermark only, unseen text,
                    results/ce_ac_{f5tts,maskgct,cosyvoice}_RQA_wm.json), K = 3 crops per probe utterance, random
                    offset, seed 20261027; a probe shorter than the drawn length is used whole

    development speakers : LibriSpeech test-clean, all 40 speakers x 5 utterances (threshold chosen here)
    test speakers        : the 20 held-out train-clean-100 speakers x 5 utterances of every cloning result
                           (threshold frozen, FAR / FRR reported here)

Rules (fixed before any score is seen):
    primary threshold   : EER on the development pairs (the observed score where |FAR - FRR| is smallest)
    secondary threshold : the lowest threshold with FAR <= 1 % on the development pairs (security-style; it rejects
                          more clones, so it is the OPTIMISTIC bound for a protection)
    reference           : 0.697 (De-AntiFake, ICML 2025: EER on LibriSpeech test-clean, 25 speakers x 5 utterances)
Reported on the test speakers at each threshold: FAR, FRR (speaker-bootstrap 95 % CIs, 10000 resamples), the test EER.

    python src/eval/sv_threshold_calibration.py --data_root DATA
    python src/eval/sv_threshold_calibration.py --data_root DATA --verifier ecapa
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

STRONG = ("f5tts", "maskgct", "cosyvoice")


def ns(**kw):
    a = argparse.Namespace(dataset="librispeech", vctk_root=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def utterances(args):
    ds = ce.load_dataset(args, crop=20.0)
    out = []
    for i in range(len(ds)):
        it = ds[i]
        out.append({"index": i, "speaker": str(it["speaker_id"]), "utterance_id": ce.utterance_id(ds, i, it),
                    "full": ce.trim_padding(it["waveform"].reshape(-1)).float().cpu()})
    return out


def clone_lengths(results_dir):
    L = []
    for cl in STRONG:
        f = os.path.join(results_dir, f"ce_ac_{cl}_RQA_wm.json")
        assert os.path.exists(f), f"{f} missing (notebook 26 result needed for the probe lengths)"
        L += [r["clone_seconds"] for r in json.load(open(f))["records"]
              if r["status"] == "ok" and r["text_mode"] == "unseen" and r.get("clone_seconds")]
    assert len(L) >= 250, len(L)
    return np.array(L)


def probe_crops(utts, lengths, k, rng):
    """K crops per utterance: (utt_pos, crop_id, waveform)."""
    out = []
    for p, u in enumerate(utts):
        x = u["full"]
        for c in range(k):
            n = int(round(float(rng.choice(lengths)) * ce.SR))
            if x.shape[-1] <= n:
                seg = x
            else:
                s0 = int(rng.integers(0, x.shape[-1] - n + 1))
                seg = x[s0:s0 + n]
            out.append((p, c, seg))
    return out


class EcapaEmbedder:
    """clone_eval_v2.Ecapa as an embedder: the same model and the same encode_batch that produced sim_ecapa."""

    def __init__(self, device):
        self.e = ce.Ecapa(device)
        self.device = device
        self._cache = {}

    def embed(self, wav, key=None):
        if key is not None and key in self._cache:
            return self._cache[key]
        with torch.no_grad():
            v = self.e.m.encode_batch(wav.reshape(1, -1).to(self.device)).reshape(-1).float().cpu().numpy()
        if key is not None:
            self._cache[key] = v
        return v


def make_verifier(name, device):
    return ce.Resem(device) if name == "resemblyzer" else EcapaEmbedder(device)


def score_matrix(resem, utts, crops, tag):
    E = np.stack([resem.embed(u["full"], key=(tag, "enrol", u["index"])) for u in utts])
    P = np.stack([resem.embed(seg) for _p, _c, seg in crops])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    return E @ P.T                                                    # [n_utts, n_crops]


def pairs(utts, crops, S):
    """genuine / impostor score lists with the enrolment speaker of each score (self-pairs excluded)."""
    spk = [u["speaker"] for u in utts]
    gen, imp = [], []
    for e in range(len(utts)):
        for j, (p, _c, _seg) in enumerate(crops):
            if p == e:
                continue                                              # never an utterance against itself
            (gen if spk[p] == spk[e] else imp).append((spk[e], spk[p], float(S[e, j])))
    return gen, imp


def far_frr(gen, imp, thr):
    g = np.array([s for _a, _b, s in gen]); i = np.array([s for _a, _b, s in imp])
    return float(np.mean(i >= thr)), float(np.mean(g < thr))


def eer(gen, imp):
    """threshold among the observed scores where |FAR - FRR| is smallest; EER = (FAR + FRR) / 2 there."""
    g = np.sort(np.array([s for _a, _b, s in gen])); i = np.sort(np.array([s for _a, _b, s in imp]))
    cand = np.unique(np.concatenate([g, i]))
    far = 1.0 - np.searchsorted(i, cand, side="left") / len(i)        # share of impostors >= t
    frr = np.searchsorted(g, cand, side="left") / len(g)              # share of genuine < t
    k = int(np.argmin(np.abs(far - frr)))
    return float(cand[k]), float((far[k] + frr[k]) / 2)


def thr_at_far(imp, target):
    """lowest threshold with FAR = mean(impostor >= t) <= target."""
    i = np.sort(np.array([s for _a, _b, s in imp]))
    allowed = int(np.floor(target * len(i)))
    t = float(np.nextafter(i[len(i) - allowed - 1], np.inf))
    assert np.mean(i >= t) <= target
    return t


def boot(gen, imp, thr, B=10000, seed=20261026):
    spks = sorted({a for a, _b, _s in gen} | {a for a, _b, _s in imp})
    G = {s: np.array([x for a, _b, x in gen if a == s]) for s in spks}
    I = {s: np.array([x for a, _b, x in imp if a == s]) for s in spks}
    rng = np.random.default_rng(seed)
    fa, fr = [], []
    for _ in range(B):
        pick = [spks[j] for j in rng.integers(0, len(spks), len(spks))]
        g = np.concatenate([G[s] for s in pick]); i = np.concatenate([I[s] for s in pick])
        fa.append(np.mean(i >= thr)); fr.append(np.mean(g < thr))
    return [float(x) for x in np.percentile(fa, [2.5, 97.5])], [float(x) for x in np.percentile(fr, [2.5, 97.5])]


def hist(gen, imp, bins=np.linspace(-0.2, 1.0, 121)):
    g = np.array([s for _a, _b, s in gen]); i = np.array([s for _a, _b, s in imp])
    return {"edges": bins.round(3).tolist(), "genuine": np.histogram(g, bins)[0].tolist(), "impostor": np.histogram(i, bins)[0].tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--verifier", choices=("resemblyzer", "ecapa"), default="resemblyzer")
    p.add_argument("--out", default=None, help="default: results/sv_calibration_{verifier}.json")
    p.add_argument("--k_crops", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--listen_indices", default="23,1,87,31,12,54",
                   help="test clips whose genuine and impostor reference crops are saved for the listening page")
    p.add_argument("--listen_dir", default="results/listen/reference")
    a = p.parse_args()
    if a.out is None:
        a.out = f"results/sv_calibration_{a.verifier}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    dev = utterances(ns(data_root=a.data_root, libri_subset="test-clean", n_speakers=0, utterances_per_speaker=0,
                        n_eval_speakers=40, eval_utterances_per_speaker=5))
    test = utterances(ns(data_root=a.data_root, libri_subset="train-clean-100", n_speakers=60, utterances_per_speaker=15,
                         n_eval_speakers=20, eval_utterances_per_speaker=5))
    test = test[:100]
    assert not ({u["speaker"] for u in dev} & {u["speaker"] for u in test}), "development and test speakers overlap"
    print(f"development: {len(dev)} utterances, {len({u['speaker'] for u in dev})} speakers | "
          f"test: {len(test)} utterances, {len({u['speaker'] for u in test})} speakers", flush=True)
    lengths = clone_lengths(a.results_dir)
    print(f"probe lengths from {len(lengths)} clones: median {np.median(lengths):.2f} s, 5-95 % "
          f"{np.percentile(lengths, 5):.2f}-{np.percentile(lengths, 95):.2f} s", flush=True)

    resem = make_verifier(a.verifier, device)
    rng = np.random.default_rng(a.seed)
    res = {"script": "sv_threshold_calibration", "settings": vars(a), "verifier": a.verifier,
           "verifier_impl": {"resemblyzer": "clone_eval_v2.Resem", "ecapa": "clone_eval_v2.Ecapa"}[a.verifier],
           "in_loss": a.verifier == "ecapa",
           "device": device, "probe_length_seconds": {"median": float(np.median(lengths)), "n_clones": int(len(lengths))}}
    sets = {}
    for name, utts in (("development", dev), ("test", test)):
        crops = probe_crops(utts, lengths, a.k_crops, rng)
        S = score_matrix(resem, utts, crops, name)
        gen, imp = pairs(utts, crops, S)
        sets[name] = (utts, crops, S, gen, imp)
        t_eer, e = eer(gen, imp)
        res[name] = {"n_genuine": len(gen), "n_impostor": len(imp), "eer": e, "eer_threshold": t_eer,
                     "genuine_mean": float(np.mean([s for *_x, s in gen])), "impostor_mean": float(np.mean([s for *_x, s in imp])),
                     "hist": hist(gen, imp)}
        print(f"{name}: {len(gen)} genuine / {len(imp)} impostor pairs | genuine mean {res[name]['genuine_mean']:.3f}, "
              f"impostor mean {res[name]['impostor_mean']:.3f} | EER {e:.3f} at {t_eer:.3f}", flush=True)

    _u, _c, _S, dg, di = sets["development"]
    thr = {"eer_dev": res["development"]["eer_threshold"], "far1_dev": thr_at_far(di, 0.01)}
    if a.verifier == "resemblyzer":
        thr["deantifake_0697"] = 0.697          # De-AntiFake (ICML 2025): Resemblyzer EER, LibriSpeech test-clean
    else:
        thr["safespeech_025"] = 0.25            # SafeSpeech (USENIX Sec. 2025): ECAPA SIM > 0.25 counts as the
        #                                         same speaker. Published criterion, not calibrated on our data,
        #                                         and used there on fine-tuned-model output: a reference, not our rule.
    res["thresholds"] = thr
    res["primary"], res["secondary"] = "eer_dev", "far1_dev"
    _u, _c, _S, tg, ti = sets["test"]
    res["test_at_thresholds"] = {}
    for k, t in thr.items():
        fa, fr = far_frr(tg, ti, t)
        ca, cr = boot(tg, ti, t)
        dfa, dfr = far_frr(dg, di, t)
        res["test_at_thresholds"][k] = {"threshold": t, "far": fa, "far_ci": ca, "frr": fr, "frr_ci": cr, "dev_far": dfa, "dev_frr": dfr}
        print(f"threshold {k:16s} {t:.3f} | dev FAR {dfa:.3f} FRR {dfr:.3f} | test FAR {fa:.3f} [{ca[0]:.3f},{ca[1]:.3f}] "
              f"FRR {fr:.3f} [{cr[0]:.3f},{cr[1]:.3f}]", flush=True)

    # reference crops for the listening page: for each listening clip, the genuine and impostor crop of the median score
    keep = [int(x) for x in a.listen_indices.split(",") if x.strip()] if a.verifier == "resemblyzer" else []
    if keep:
        import soundfile as sf
        os.makedirs(a.listen_dir, exist_ok=True)
        utts, crops, S, _g, _i = sets["test"]
        ref = {}
        for e in keep:
            row = {"genuine": [], "impostor": []}
            for j, (pp, _c, seg) in enumerate(crops):
                if pp == e:
                    continue
                row["genuine" if utts[pp]["speaker"] == utts[e]["speaker"] else "impostor"].append((float(S[e, j]), j))
            ref[str(e)] = {}
            for kind in ("genuine", "impostor"):
                sc = sorted(row[kind])
                s, j = sc[len(sc) // 2]
                pp = crops[j][0]
                path = os.path.join(a.listen_dir, f"{e}_{kind}.wav")
                sf.write(path, crops[j][2].numpy(), ce.SR)
                ref[str(e)][kind] = {"score": s, "probe_utterance": utts[pp]["utterance_id"], "probe_speaker": utts[pp]["speaker"],
                                     "file": path, "score_range": [sc[0][0], sc[-1][0]]}
            sf.write(os.path.join(a.listen_dir, f"{e}_enrolment.wav"), utts[e]["full"].numpy(), ce.SR)
        res["listening_reference"] = ref
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out, "|", res["seconds"], "s")


if __name__ == "__main__":
    main()
