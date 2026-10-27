"""
src/eval/prompt_asr_eval.py  (2026-10-26, docs/preregistration_q2t_2026-10-26.md, part B, prompt level)

What an attacker's recogniser hears in each protected PROMPT (before any cloning). Three recognisers:
    base_en   faster-whisper base.en     (the prompt transcriber clone_eval_v2 gives MaskGCT and CosyVoice;
                                          same weights as the q2t surrogate: matched)
    turbo     faster-whisper large-v3-turbo (what F5-TTS transcribes with: held out)
    w2v2      wav2vec2-base-960h, greedy CTC (another family: held out)
Reference = the turbo transcript of the CLEAN prompt. WER of every arm's transcript against it; for q2t / t also
against the target sentence (from records_ARM). Arms are whatever of clean, wm, rs, q2, a0, q2c, t, q2t exist.

    python src/eval/prompt_asr_eval.py --prompt_dir results/anticlone/prompts/RQA --out results/anticlone/prompt_asr_RQA.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "models", "data", "losses"):
    sys.path.insert(0, os.path.join(HERE, "..", sub) if sub else os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import clone_eval_v2 as ce  # noqa: E402

ARMS = ("clean", "wm", "rs", "q2", "a0", "q2c", "t", "q2t")


class FW:
    def __init__(self, size, device):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(size, device=device, compute_type="float16" if device == "cuda" else "int8")

    def __call__(self, x):
        segs, _ = self.m.transcribe(np.ascontiguousarray(x, dtype=np.float32), language="en", beam_size=5)
        return " ".join(s.text for s in segs).strip()


class W2V:
    def __init__(self, device):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.p = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.m = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
        self.d = device

    @torch.no_grad()
    def __call__(self, x):
        v = self.p(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.d)
        ids = self.m(v).logits.argmax(-1)
        return self.p.batch_decode(ids)[0].lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    a = p.parse_args()
    import soundfile as sf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = [x for x in ARMS if os.path.isdir(os.path.join(a.prompt_dir, x))
            and all(os.path.exists(os.path.join(a.prompt_dir, x, f"{i}.flac")) for i in range(a.n))]
    print("arms:", arms, flush=True)
    targets = {}
    for arm in ("t", "q2t"):
        m = os.path.join(a.prompt_dir, f"{arm}_manifest.json")
        if os.path.exists(m):
            targets[arm] = {r["index"]: r["asr"]["target"] for r in json.load(open(m))["records"]}
    asrs = {"base_en": FW("base.en", device), "turbo": FW("large-v3-turbo", device), "w2v2": W2V(device)}
    out = {"script": "prompt_asr_eval", "arms": arms, "reference": "turbo transcript of the clean prompt", "records": []}
    for i in range(a.n):
        r = {"index": i, "heard": {}, "wer": {}, "wer_vs_target": {}}
        for arm in arms:
            x, sr = sf.read(os.path.join(a.prompt_dir, arm, f"{i}.flac"), dtype="float32")
            assert sr == 16000
            r["heard"][arm] = {k: f(x) for k, f in asrs.items()}
        ref = r["heard"]["clean"]["turbo"]
        r["reference"] = ref
        for arm in arms:
            r["wer"][arm] = {k: ce.wer(ref, h) for k, h in r["heard"][arm].items()}
            if arm in targets:
                r["wer_vs_target"][arm] = {k: ce.wer(targets[arm][i], h) for k, h in r["heard"][arm].items()}
        out["records"].append(r)
        print(f"[{i}] " + " | ".join(f"{arm} {r['wer'][arm]['base_en']:.2f}/{r['wer'][arm]['turbo']:.2f}" for arm in arms), flush=True)
    summ = {}
    for arm in arms:
        summ[arm] = {k: float(np.nanmean([np.nan if r["wer"][arm][k] is None else r["wer"][arm][k] for r in out["records"]]))
                     for k in asrs}
        if arm in targets:
            summ[arm]["vs_target"] = {k: float(np.nanmean([np.nan if r["wer_vs_target"][arm][k] is None else r["wer_vs_target"][arm][k]
                                                            for r in out["records"]])) for k in asrs}
    out["summary_mean_wer"] = summ
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("saved", a.out)


if __name__ == "__main__":
    main()
