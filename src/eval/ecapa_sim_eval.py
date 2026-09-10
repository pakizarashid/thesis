"""
src/eval/ecapa_sim_eval.py

Makes this project's speaker-similarity numbers COMPARABLE TO SafeSpeech's.

THE PROBLEM THIS SOLVES
------------------------
Every SIM value in this repo is computed with YourTTS's own speaker encoder
(`surrogate.compute_speaker_embedding`). SafeSpeech computes SIM with **ECAPA-TDNN**,
and their success threshold is defined for that encoder:

    "We follow the principles from [25] and leverage ECAPA-TDNN [13] as the speaker
     encoder to compute the cosine similarity score between the real and synthetic
     speeches. When the SIM exceeds 0.25, personal voice has been successfully cloned
     in timbre. The Attack Success Rate (ASR) is calculated by the ratio of
     successfully cloned samples in speaker similarity (i.e., SIM > 0.25) to the
     total number of samples."

Cosine similarity is not comparable across embedding spaces. A 0.25 cutoff calibrated
on ECAPA-TDNN says nothing about a YourTTS-encoder cosine, and quoting this project's
0.2664 against SafeSpeech's 0.284 is invalid. Until SIM is recomputed with ECAPA-TDNN,
no claim of the form "our protection is better/comparable to SafeSpeech" can be made.

Note also that SafeSpeech reports **ASR** (fraction of clones above threshold), not the
mean SIM. Means are misleading here: this project's protected-clone distribution is
right-skewed, so a handful of high samples pull the mean above 0.25 while the median
sits near 0.21 and most clones fail. This script reports mean, median AND ASR.

WHY IT IS A SEPARATE SCRIPT (DEPENDENCY ISOLATION)
---------------------------------------------------
`pip install denoiser` already broke this environment once by downgrading omegaconf
below 2.0, after which the VoiceMark checkpoint could no longer be unpickled. speechbrain
pulls its own dependency set and carries the same risk. So ECAPA scoring runs as a
SEPARATE process over saved WAV files: nothing here imports the backbone, the surrogate,
or any VoiceMark code. If installing speechbrain damages something, only this script is
affected and the main pipeline keeps working.

    pip install speechbrain

INPUT LAYOUT
------------
A directory of WAV pairs. For each sample index i, a reference and one or more clones:

    <dir>/sample{i}_reference.wav        the real speaker's audio (ground truth)
    <dir>/sample{i}_<condition>.wav      a clone to score against it

Every condition found is scored separately, so one run compares all of them:

    sample0_reference.wav
    sample0_clone_clean.wav        -> unprotected baseline
    sample0_clone_protected.wav    -> protection working
    sample0_clone_demucs.wav       -> protection after attack

Usage:
    python src/eval/ecapa_sim_eval.py --sample_dir ./audio_samples/demucs_fallback \\
        --output results/results_ecapa_sim.json
"""

import os
import re
import glob
import json
import argparse


def load_ecapa(device):
    """
    ECAPA-TDNN from speechbrain (spkrec-ecapa-voxceleb) -- the same model family
    SafeSpeech uses, so its cosine scale and the 0.25 threshold apply.
    """
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return model


def embed(model, path, device):
    import torchaudio
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with __import__("torch").no_grad():
        emb = model.encode_batch(wav.to(device)).squeeze()
    return emb


def main():
    import torch
    import torch.nn.functional as F

    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.25,
                   help="SafeSpeech's ECAPA-TDNN success threshold. Do NOT change this -- "
                        "it is what makes the ASR figure comparable to their table.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_ecapa(device)
    print(f"[ecapa] loaded speechbrain/spkrec-ecapa-voxceleb on {device}")

    refs = sorted(glob.glob(os.path.join(args.sample_dir, "sample*_reference.wav")))
    if not refs:
        raise SystemExit(
            f"No sample*_reference.wav in {args.sample_dir}. Generate the WAVs first "
            f"(demucs_fallback_eval.py --save_clones_dir ...)."
        )
    print(f"[ecapa] found {len(refs)} reference files")

    per_condition = {}
    for ref_path in refs:
        idx = re.search(r"sample(\d+)_reference\.wav$", os.path.basename(ref_path)).group(1)
        emb_ref = embed(model, ref_path, device)
        for clone_path in sorted(glob.glob(os.path.join(args.sample_dir, f"sample{idx}_*.wav"))):
            base = os.path.basename(clone_path)
            cond = base[len(f"sample{idx}_"):-len(".wav")]
            if cond == "reference":
                continue
            emb_c = embed(model, clone_path, device)
            sim = F.cosine_similarity(emb_ref.unsqueeze(0), emb_c.unsqueeze(0)).item()
            per_condition.setdefault(cond, []).append(sim)

    print(f"\n{'=' * 78}")
    print(f"ECAPA-TDNN SPEAKER SIMILARITY  (comparable to SafeSpeech; threshold {args.threshold})")
    print(f"{'=' * 78}")
    print(f"{'condition':<26}{'mean':>9}{'median':>9}{'ASR':>9}   {'n':>4}")
    results = {}
    for cond, vals in per_condition.items():
        v = sorted(vals)
        n = len(v)
        mean = sum(v) / n
        median = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
        asr = sum(1 for x in v if x > args.threshold) / n
        results[cond] = {"mean": mean, "median": median, "asr": asr, "n": n, "values": vals}
        print(f"{cond:<26}{mean:>9.4f}{median:>9.4f}{asr * 100:>8.1f}%{n:>5}")

    print(f"\nASR = fraction of clones exceeding {args.threshold} -- SafeSpeech's own success")
    print(f"criterion. This, not the mean, is the number to compare against their table.")
    print(f"SafeSpeech reference points (ECAPA-TDNN): protected SIM 0.204, after DEMUCS 0.284,")
    print(f"after MP3 0.261, unprotected 0.604.")
    print(f"\nREMAINING CAVEAT: same encoder now, but still different corpus, different cloning")
    print(f"models and a different protection implementation. Closer, not identical --")
    print(f"state that in the write-up.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"label": "ecapa_sim", "sample_dir": args.sample_dir,
                       "encoder": "speechbrain/spkrec-ecapa-voxceleb",
                       "threshold": args.threshold, "results": results}, f, indent=2)
        print(f"\n[main] Saved to {args.output}")


if __name__ == "__main__":
    main()
