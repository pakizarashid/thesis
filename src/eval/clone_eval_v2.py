"""
src/eval/clone_eval_v2.py

Watermark survival through zero-shot cloning: ONE protocol for every cloner
(YourTTS, XTTS-v2, F5-TTS, MaskGCT, CosyVoice, FishSpeech). Written 2026-10-24.

The cloner backends are the ones in cloner_watermark_eval.py (imported, not
copied), plus a YourTTS backend that uses the same surrogate_vc.clone_voice
path as watermark_survival_under_cloning.py. Backends are imported lazily, so
this file runs in the coqui environment (YourTTS, XTTS-v2) and in each separate
cloner environment (F5-TTS, MaskGCT, CosyVoice) without the others installed.

What this adds over the older scripts
-------------------------------------
- Per-clip records with the utterance ID, speaker, prompt length actually used,
  target text, prompt transcript, ACC, bits correct, traced, FAR, clone length.
  Older YourTTS results stored means only and older XTTS results stored values
  without IDs, so pairing across files was assumed, not checked.
- Prompt length (--prompts 3,6,full): every prompt is a centre crop of the
  same utterance; 'full' is the whole utterance (trailing padding removed,
  capped at --max_full_seconds). The 3 s prompt is identical to the crop the
  older scripts used.
- Text protocol (--texts fixed,unseen):
    fixed  = "This is a test sentence for voice cloning." (all earlier results)
    unseen = per clip, a transcript of ANOTHER held-out speaker's utterance
             (6-20 words), never used by any training run. Route 2 samples its
             clone texts from TRAINING transcripts only; pass that pool with
             --train_text_pool and the overlap is checked and recorded.
- Traced = at least 13 of 16 bits correct (project definition).
- FAR, as in VoiceMark: the decoded message is compared by Hamming distance with
  the true message and with 99 random candidates (seed far_seed + clip). A false
  attribution is any candidate at least as close as the truth (ties count as
  false: conservative).
- Optional ECAPA-TDNN similarity (--ecapa) of the clone to the CLEAN full
  utterance, and WER (--wer, faster-whisper) of the clone against its target
  text, so the cloner-works gate (SIM > 0.30 and WER < 20%) is computed in the
  same run.
- Provenance: SHA-256 of the checkpoint and the VoiceMark weights, git commit,
  package versions, written into the output.
- Resumable: if --output exists for the same cloner and checkpoint hash, done
  records are kept and only missing ones are run. Partial results are written
  every few clones.

Messages: clip i carries 16 bits drawn with seed --message_seed_base + i (123 + i
by default), the same for every prompt length, text and checkpoint.

Usage (coqui environment):
    python src/eval/clone_eval_v2.py --cloner xtts --checkpoint CKPT \\
        --prompts 3 --texts fixed,unseen --ecapa --wer --output results/x.json
Usage (MaskGCT environment):
    python src/eval/clone_eval_v2.py --cloner maskgct --amphion_root /kaggle/working/Amphion \\
        --checkpoint CKPT --prompts 3 --texts fixed,unseen --ecapa --wer --output results/m.json
"""

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("", "models", "data", "losses"):
    sys.path.insert(0, os.path.join(HERE, "..", sub) if sub else os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import cloner_watermark_eval as cwe  # noqa: E402  (backends are lazy; no cloner imported here)

FIXED_TEXT = "This is a test sentence for voice cloning."
SR = 16000


# ----------------------------------------------------------------------------- provenance
def sha256(path, chunk=1 << 20):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_info():
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("git rev-parse HEAD"), "branch": run("git rev-parse --abbrev-ref HEAD"),
            "dirty_files": [l for l in (run("git status --porcelain") or "").splitlines() if l][:50]}


def package_versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for p in ("torch", "torchaudio", "transformers", "coqui-tts", "f5-tts", "speechtokenizer",
              "encodec", "numpy", "faster-whisper", "speechbrain", "huggingface-hub"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            pass
    out["python"] = platform.python_version()
    return out


# ----------------------------------------------------------------------------- text
def normalize_text(t):
    t = " ".join(str(t).strip().split())
    if not t:
        return t
    if t.isupper():
        t = t.lower()
    t = t[0].upper() + t[1:]
    if t[-1] not in ".?!":
        t += "."
    return t


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def build_unseen_texts(meta, min_words, max_words):
    """meta: list of (speaker, transcript) per clip. Clip i gets the transcript of a
    clip from a DIFFERENT speaker, chosen deterministically, 6-20 words.
    Returns (texts, source_clip_index)."""
    pool = [(j, spk, normalize_text(t)) for j, (spk, t) in enumerate(meta)
            if t and min_words <= len(words(t)) <= max_words]
    texts, sources = [], []
    for i, (spk, _) in enumerate(meta):
        cands = [(j, t) for j, s, t in pool if s != spk]
        if not cands:
            raise SystemExit("no unseen-text candidates from other speakers")
        j, t = cands[(i * 7 + 3) % len(cands)]
        assert meta[j][0] != spk
        texts.append(t)
        sources.append(j)
    return texts, sources


# ----------------------------------------------------------------------------- data
def load_dataset(args, crop):
    if args.dataset == "librispeech":
        from librispeech import LibriSpeechSubset
        return LibriSpeechSubset(root=args.data_root, n_speakers=args.n_speakers,
                                 utterances_per_speaker=args.utterances_per_speaker,
                                 n_eval_speakers=args.n_eval_speakers,
                                 eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                                 sample_rate=SR, crop_seconds=crop, split="eval",
                                 subset=args.libri_subset)
    from vctk import VCTKSubset
    return VCTKSubset(root=args.vctk_root, n_speakers=args.n_speakers,
                      utterances_per_speaker=args.utterances_per_speaker,
                      n_eval_speakers=args.n_eval_speakers,
                      eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                      sample_rate=SR, crop_seconds=crop, split="eval")


def utterance_id(ds, i, item):
    if "utterance_id" in item:
        return item["utterance_id"]
    try:
        return ds._items[i]["utterance_id"]
    except Exception:
        return str(i)


def trim_padding(w, eps=1e-4):
    nz = (w.abs() > eps).nonzero()
    return w if nz.numel() == 0 else w[: nz[-1].item() + 1]


def make_prompt(full, spec, max_full):
    if spec == "full":
        return full[: int(max_full * SR)]
    n = int(round(float(spec) * SR))
    if full.shape[-1] <= n:
        return full
    start = (full.shape[-1] - n) // 2  # centre crop, as LibriSpeechSubset/VCTKSubset do for eval
    return full[start:start + n]


# ----------------------------------------------------------------------------- detector
def message_bits(seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 2, (1, 16), generator=gen, device=device)


def decode_bits(backbone, wav):
    with torch.no_grad():
        feat = backbone.model.st_model.forward_feature(wav)
        _logits, chunk_logits = backbone.model.detector(feat)
    pred = torch.argmax(chunk_logits, dim=-1)[0]  # [4] chunk values
    return [int((int(pred[c]) >> b) & 1) for c in range(pred.shape[0]) for b in range(4)]


def far_check(dec, true, seed, n_cand):
    rng = np.random.default_rng(seed)
    dec, true = np.asarray(dec), np.asarray(true)
    d_true = int((dec != true).sum())
    cands = rng.integers(0, 2, size=(n_cand, dec.shape[0]))
    d = (cands != dec[None, :]).sum(axis=1)
    return bool((d <= d_true).any()), d_true


# ----------------------------------------------------------------------------- scorers
class Ecapa:
    def __init__(self, device):
        from speechbrain.inference.speaker import EncoderClassifier
        self.m = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                                savedir="/tmp/ecapa_model", run_opts={"device": device})

    def __call__(self, a, b):
        with torch.no_grad():
            ea = self.m.encode_batch(a.reshape(1, -1)).reshape(-1)
            eb = self.m.encode_batch(b.reshape(1, -1)).reshape(-1)
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))


class Asr:
    def __init__(self, device, size="base.en"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(size, device=device, compute_type="float16" if device == "cuda" else "int8")

    def __call__(self, wav16k):
        x = wav16k.detach().cpu().reshape(-1).numpy().astype(np.float32)
        segs, _ = self.m.transcribe(np.ascontiguousarray(x), language="en", beam_size=5)
        return " ".join(s.text for s in segs).strip()


# ----------------------------------------------------------------------------- YourTTS backend
def load_yourtts(device, args):
    from surrogate_vc import load_yourtts_surrogate
    return load_yourtts_surrogate(device=device)


def clone_yourtts(model, speaker_audio_16k, gen_text, tmp_dir, tag, ref_text, args):
    with torch.no_grad():  # same path as watermark_survival_under_cloning.py (16 kHz output)
        return model.clone_voice(speaker_audio_16k, text=gen_text).detach()


BACKENDS = dict(cwe.BACKENDS)
BACKENDS["yourtts"] = (load_yourtts, clone_yourtts, "fixed d-vector", None)
NEEDS_PROMPT_TEXT = {"maskgct", "cosyvoice", "fishspeech"}


# ----------------------------------------------------------------------------- summary
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def speaker_bootstrap(vals, spk, n_boot=2000, seed=20261024):
    vals, spk = np.asarray(vals, float), np.asarray(spk)
    groups = [vals[spk == s] for s in np.unique(spk)]
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
             for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(records):
    out = {}
    for key in sorted({(r["prompt"], r["text_mode"]) for r in records}):
        rs = [r for r in records if (r["prompt"], r["text_mode"]) == key]
        ok = [r for r in rs if r["status"] == "ok"]
        s = {"n_ok": len(ok), "n_failed": len(rs) - len(ok)}
        if ok:
            acc = [r["acc_clone"] for r in ok]
            spk = [r["speaker"] for r in ok]
            traced = int(sum(bool(r["traced"]) for r in ok))
            far = int(sum(bool(r["far_false"]) for r in ok))
            s.update(acc_source_mean=float(np.mean([r["acc_source"] for r in ok])),
                     acc_mean=float(np.mean(acc)), acc_ci_speaker=speaker_bootstrap(acc, spk),
                     traced_rate=float(traced / len(ok)), traced_ci_wilson=wilson(traced, len(ok)),
                     far_rate=float(far / len(ok)), far_ci_wilson=wilson(far, len(ok)),
                     prompt_seconds_mean=float(np.mean([r["prompt_seconds"] for r in ok])),
                     clone_seconds_mean=float(np.mean([r["clone_seconds"] for r in ok])))
            sims = [r["sim_ecapa"] for r in ok if r.get("sim_ecapa") is not None]
            wers = [r["wer"] for r in ok if r.get("wer") is not None]
            if sims:
                s.update(sim_ecapa_mean=float(np.mean(sims)), asr_sim_gt_025=float(np.mean(np.array(sims) > 0.25)))
                tr_hi = [r["traced"] for r in ok if r.get("sim_ecapa") is not None and r["sim_ecapa"] > 0.25]
                s["traced_given_sim_gt_025"] = float(np.mean(tr_hi)) if tr_hi else None
                s["attacker_wins"] = float(np.mean([(r["sim_ecapa"] > 0.25) and not r["traced"]
                                                    for r in ok if r.get("sim_ecapa") is not None]))
            if wers:
                s["wer_mean"] = float(np.mean(wers))
            if sims and wers:
                s["gate_pass"] = bool(np.mean(sims) > 0.30 and np.mean(wers) < 0.20)
        out[f"{key[0]}|{key[1]}"] = s
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cloner", required=True, choices=sorted(BACKENDS))
    p.add_argument("--checkpoint", default=None, help="Omit for pretrained VoiceMark.")
    p.add_argument("--output", required=True)
    p.add_argument("--prompts", default="3", help="Comma list of seconds and/or 'full'.")
    p.add_argument("--texts", default="fixed,unseen", help="Comma list of fixed, unseen.")
    p.add_argument("--fixed_text", default=FIXED_TEXT)
    p.add_argument("--unseen_min_words", type=int, default=6)
    p.add_argument("--unseen_max_words", type=int, default=20)
    p.add_argument("--train_text_pool", default=None,
                   help="JSON list of training clone texts (written by Route 2); overlap is checked.")
    p.add_argument("--max_full_seconds", type=float, default=20.0)
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--message_seed_base", type=int, default=123)
    p.add_argument("--far_candidates", type=int, default=99)
    p.add_argument("--far_seed", type=int, default=777)
    p.add_argument("--ecapa", action="store_true", help="ECAPA-TDNN SIM of clone vs clean utterance.")
    p.add_argument("--wer", action="store_true", help="faster-whisper WER of clone vs target text.")
    p.add_argument("--whisper_size", default="base.en")
    p.add_argument("--save_audio_dir", default=None)
    p.add_argument("--tmp_dir", default="/tmp/clone_eval_v2")
    p.add_argument("--dump_every", type=int, default=5)
    p.add_argument("--dataset", choices=("librispeech", "vctk"), default="librispeech")
    p.add_argument("--data_root", default="./data/librispeech")
    p.add_argument("--libri_subset", default="train-clean-100",
                   help="LibriSpeech part. 'test-clean' is the set VoiceMark evaluated on; use "
                        "--n_speakers 0 --n_eval_speakers 40 with it (all 40 test speakers).")
    p.add_argument("--vctk_root", default="/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    # backend options (passed through to cloner_watermark_eval backends)
    p.add_argument("--target_len", type=float, default=None)
    p.add_argument("--cosyvoice_root", default=None)
    p.add_argument("--cosyvoice_model_dir", default=None)
    p.add_argument("--amphion_root", default=None)
    p.add_argument("--fish_url", default="http://127.0.0.1:8080")
    p.add_argument("--diagnostic", action="store_true", help="2 clips, all conditions, no output file.")
    args = p.parse_args()

    cwe.preflight()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [x.strip() for x in args.prompts.split(",") if x.strip()]
    texts = [x.strip() for x in args.texts.split(",") if x.strip()]
    assert set(texts) <= {"fixed", "unseen"}, texts
    n_clips = 2 if args.diagnostic else args.n_utterances

    vm_ckpt = os.path.join(REPO, "external", "voicemark", "voicemark.pth")
    st_ckpt = os.path.join(REPO, "external", "voicemark", "speechtokenizer", "pretrained_model", "SpeechTokenizer.pt")
    prov = {"checkpoint": args.checkpoint, "checkpoint_sha256": sha256(args.checkpoint),
            "voicemark_pth_sha256": sha256(vm_ckpt), "speechtokenizer_pt_sha256": sha256(st_ckpt),
            "git": git_info(), "packages": package_versions(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        prov["checkpoint_epoch"] = ck.get("epoch")
        prov["checkpoint_train_args"] = ck.get("train_args")
        del ck

    # data: one dataset loaded long enough to hold whole utterances
    ds = load_dataset(args, crop=args.max_full_seconds)
    n_clips = min(n_clips, len(ds))
    items = []
    for i in range(n_clips):
        it = ds[i]
        full = trim_padding(it["waveform"].reshape(-1))
        items.append({"index": i, "utterance_id": utterance_id(ds, i, it), "speaker": str(it["speaker_id"]),
                      "transcript": it.get("transcript", ""), "full": full})
    unseen, unseen_src = build_unseen_texts([(it["speaker"], it["transcript"]) for it in items],
                                            args.unseen_min_words, args.unseen_max_words)
    overlap = None
    if args.train_text_pool:
        train_pool = {normalize_text(t) for t in json.load(open(args.train_text_pool))}
        overlap = sorted({t for t in unseen if t in train_pool})
        if overlap:
            raise SystemExit(f"{len(overlap)} 'unseen' texts occur in the training pool: {overlap[:3]}")
        overlap = []

    # resume
    records, done = [], set()
    if os.path.exists(args.output) and not args.diagnostic:
        old = json.load(open(args.output))
        if old.get("cloner") == args.cloner and old.get("provenance", {}).get("checkpoint_sha256") == prov["checkpoint_sha256"] \
                and old.get("args", {}).get("dataset") == args.dataset:
            records = [r for r in old.get("records", []) if r["status"] == "ok"]
            done = {(r["index"], r["prompt"], r["text_mode"]) for r in records}
            print(f"[resume] {len(records)} completed records kept from {args.output}")
        else:
            raise SystemExit(f"{args.output} exists for a different cloner/checkpoint/dataset; refusing to overwrite")

    def dump(final=False):
        if args.diagnostic:
            return
        out = {"script": "clone_eval_v2", "cloner": args.cloner, "args": vars(args), "provenance": prov,
               "unseen_texts": {str(it["index"]): {"text": unseen[it["index"]], "from_clip": unseen_src[it["index"]]}
                                for it in items},
               "clips": [{k: it[k] for k in ("index", "utterance_id", "speaker", "transcript")} for it in items],
               "train_text_overlap": overlap, "records": records, "summary": summarise(records),
               "complete": final}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, args.output)

    todo = [(it, sp, tm) for it in items for sp in prompts for tm in texts if (it["index"], sp, tm) not in done]
    print(f"[plan] cloner={args.cloner} clips={n_clips} prompts={prompts} texts={texts} -> {len(todo)} clones to run")
    if not todo:
        dump(final=True)
        print("[done] nothing left to run")
        return

    backbone = cwe.build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    load_fn, clone_fn, mechanism, published = BACKENDS[args.cloner]
    model = load_fn(device, args)
    transcribe = cwe.Transcriber(args.whisper_size, device) if args.cloner in NEEDS_PROMPT_TEXT else None
    ecapa = Ecapa(device) if args.ecapa else None
    asr = Asr(device, args.whisper_size) if args.wer else None
    os.makedirs(args.tmp_dir, exist_ok=True)

    cache = {}  # (index, prompt) -> (recon_wm, msg, prompt_seconds, acc_source, ref_text)
    for k, (it, spec, mode) in enumerate(todo):
        i = it["index"]
        msg = message_bits(args.message_seed_base + i, device)
        true_bits = msg[0].tolist()
        if (i, spec) not in cache:
            prompt = make_prompt(it["full"], spec, args.max_full_seconds).to(device).reshape(1, 1, -1)
            with torch.no_grad():
                recon_wm = backbone.forward_full(prompt, msg)["recon_wm"]
            src_bits = decode_bits(backbone, recon_wm)
            acc_src = float(np.mean(np.array(src_bits) == np.array(true_bits)))
            ref_text = ""
            if transcribe is not None:
                ref_text = transcribe(cwe.write_ref_wav(recon_wm, args.tmp_dir, f"tr{i}_{spec}"))
            cache = {(i, spec): (recon_wm, prompt.shape[-1] / SR, acc_src, ref_text)}  # keep one entry
        recon_wm, prompt_sec, acc_src, ref_text = cache[(i, spec)]
        gen_text = args.fixed_text if mode == "fixed" else unseen[i]
        rec = {"index": i, "utterance_id": it["utterance_id"], "speaker": it["speaker"], "prompt": spec,
               "prompt_seconds": round(prompt_sec, 3), "text_mode": mode, "gen_text": gen_text,
               "ref_text": ref_text, "acc_source": acc_src, "status": "ok"}
        try:
            cloned = clone_fn(model, recon_wm, gen_text, args.tmp_dir, f"c{i}_{spec}_{mode}", ref_text, args)
            cloned = cloned.reshape(1, 1, -1).to(device)
            dec = decode_bits(backbone, cloned)
            correct = int((np.array(dec) == np.array(true_bits)).sum())
            far_false, d_true = far_check(dec, true_bits, args.far_seed + i, args.far_candidates)
            rec.update(acc_clone=correct / 16, bits_correct=correct, traced=correct >= 13,
                       far_false=far_false, hamming_true=d_true, decoded_bits="".join(map(str, dec)),
                       clone_seconds=round(cloned.shape[-1] / SR, 3))
            if ecapa is not None:
                rec["sim_ecapa"] = ecapa(cloned.reshape(-1), it["full"].to(device))
            if asr is not None:
                hyp = asr(cloned)
                rec["asr_text"], rec["wer"] = hyp, wer(gen_text, hyp)
            if args.save_audio_dir:
                import soundfile as sf
                d = os.path.join(args.save_audio_dir, f"p{spec}_{mode}")
                os.makedirs(d, exist_ok=True)
                sf.write(os.path.join(d, f"sample{i}_watermarked.wav"), recon_wm.reshape(-1).cpu().numpy(), SR)
                sf.write(os.path.join(d, f"sample{i}_clone_{args.cloner}.wav"), cloned.reshape(-1).cpu().numpy(), SR)
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        records.append(rec)
        line = (f"[{k + 1}/{len(todo)}] clip {i} prompt {spec} text {mode}: "
                + (f"acc_src={acc_src:.3f} acc={rec['acc_clone']:.3f} traced={rec['traced']} far={rec['far_false']}"
                   + (f" sim={rec['sim_ecapa']:.3f}" if rec.get("sim_ecapa") is not None else "")
                   + (f" wer={rec['wer']:.2f}" if rec.get("wer") is not None else "")
                   if rec["status"] == "ok" else f"FAILED {rec['error']}"))
        print(line, flush=True)
        if args.diagnostic and k == 0 and rec["status"] == "ok" and acc_src < 0.85:
            raise SystemExit(f"source ACC {acc_src:.3f} on the watermarked prompt: the mark is not there")
        if (k + 1) % args.dump_every == 0:
            dump()

    dump(final=True)
    print("\n=== summary (prompt|text) ===")
    for key, s in summarise(records).items():
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                    if k in ("n_ok", "n_failed", "acc_source_mean", "acc_mean", "traced_rate", "far_rate",
                             "sim_ecapa_mean", "wer_mean", "gate_pass")})
    if not args.diagnostic:
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
