"""
src/eval/protect_prompts_q2c.py  (anti-cloning: intelligibility term, 2026-10-26)

Arm q2c = Q2 plus a term that makes the clone harder to understand. Everything in Q2 is unchanged (same bound,
steps, seeds, verifiers, F5-TTS surrogate rollout); the extra term is the CTC loss of wav2vec2-base-960h
(facebook/wav2vec2-base-960h) on the rollout against the PGD sentence, which the attack maximises. Its gradient is
rescaled to --content_weight x ||grad_wm|| (default 1.732, the same weight as the speaker term), see
f5_whitebox_pgd.whitebox_pgd(content_fn=...).

The recogniser in the loss (wav2vec2, CTC) is a different model family from the one that scores WER
(faster-whisper base.en in clone_eval_v2.py), so the WER result is not measured by the model that was attacked.

Same clips, crop, messages and clean-prompt hash check as protect_prompts_v2.py (its manifest is reused); output
OUT/q2c/{i}.flac (16-bit), OUT/records_q2c/{i}.json, and --merge writes OUT/q2c_manifest.json. F5-TTS environment.

    python src/eval/protect_prompts_q2c.py --checkpoint CKPT --out_dir results/anticlone/prompts/RQA DATA_ARGS --start 0 --stop 50
    python src/eval/protect_prompts_q2c.py ... --merge
    python src/eval/protect_prompts_q2c.py ... --diagnostic     (clip 0, verbose, writes nothing)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "models", "data", "losses"):
    sys.path.insert(0, os.path.join(HERE, "..", sub) if sub else os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import clone_eval_v2 as ce  # noqa: E402
import cloner_watermark_eval as cwe  # noqa: E402
from protect_prompts_v2 import Transcriber, quality, read_audio, wav_sha, write_flac  # noqa: E402


class CtcContentLoss:
    """wav2vec2-base-960h CTC loss of a waveform against a fixed text (upper-case letters, '|' word separator)."""
    def __init__(self, device, text):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        clean = "".join(ch for ch in text.upper() if ch.isalpha() or ch in " '").strip()
        ids = self.proc.tokenizer(clean).input_ids
        self.target = torch.tensor(ids, device=device).unsqueeze(0)
        self.blank = self.proc.tokenizer.pad_token_id
        assert self.target.numel() > 0, "empty CTC target"

        # frames a CTC alignment of the target needs: one per label plus a blank between repeated labels
        t = self.target[0].tolist()
        self.min_frames = len(t) + sum(1 for a, b in zip(t, t[1:]) if a == b)
        self.n_short = 0

    def __call__(self, wave, sr):
        x = wave.reshape(1, -1).float()
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)
        x = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-7)
        logp = F.log_softmax(self.model(x).logits.float(), dim=-1).transpose(0, 1)    # [T, 1, V]
        T = logp.shape[0]
        if T < self.min_frames:
            # rollout too short for any alignment of the target: CTC is +inf (its CUDA backward gave NaN, 2026-10-26).
            # No content term on this step (whitebox_pgd then takes the plain Q2 step and counts it).
            self.n_short += 1
            return torch.zeros((), device=x.device)
        # CTC in float64 on the CPU: the CPU kernel's zero_infinity handling is exact; the gradient returns to the GPU
        return F.ctc_loss(logp.double().cpu(), self.target.cpu(), torch.tensor([T]), torch.tensor([self.target.shape[1]]),
                          blank=self.blank, reduction="mean", zero_infinity=True).to(x.device).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out_dir", default=None, help="the protect_prompts_v2.py output dir of the same checkpoint")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    p.add_argument("--content_weight", type=float, default=1.732)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
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
    args = p.parse_args()
    if not args.out_dir:
        p.error("--out_dir is required (its manifest.json gives Q2's exact settings and the clean-prompt hashes)")

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.load(open(os.path.join(args.out_dir, "manifest.json")))
    assert man["complete"], "run protect_prompts_v2.py (and its --merge) for this checkpoint first"
    ck_sha = ce.sha256(args.checkpoint)
    assert man["settings"]["checkpoint_sha256"] == ck_sha, "--checkpoint differs from the one the other arms were made with"
    q2 = man["settings"]                              # Q2's registered settings, reused verbatim
    base = {r["index"]: r for r in man["records"]}
    n_clips = min(args.n_utterances, man["n_clips"])
    rec_dir = os.path.join(args.out_dir, "records_q2c")

    if args.merge:
        recs = [json.load(open(os.path.join(rec_dir, f"{i}.json"))) for i in range(n_clips)
                if os.path.exists(os.path.join(rec_dir, f"{i}.json"))]
        missing = sorted(set(range(n_clips)) - {r["index"] for r in recs})
        summ = {k: float(np.mean([r["q2c"][k] for r in recs])) for k in ("acc_source", "pesq", "stoi", "si_snr", "snr_db")} if recs else {}
        summ["clips_over_10_skipped"] = [r["index"] for r in recs if r["diag"]["skipped"] > 10]
        summ["clips_content_unusable_over_half"] = [r["index"] for r in recs
                                                    if r["diag"].get("ct_nonfinite_steps", 0) + r["diag"].get("ct_zero_steps", 0) > 20]
        out = {"script": "protect_prompts_q2c", "settings": {**{k: v for k, v in vars(args).items() if k not in ("start", "stop", "merge", "diagnostic")},
                                                              "q2_settings": {k: q2[k] for k in ("epsilon", "shape_alpha", "shape_floor", "n_steps", "step_size",
                                                                                                  "dis_to_wm_ratio", "lambda_wm", "f5_steps", "f5_cfg", "pgd_text")}},
               "checkpoint_sha256": ck_sha, "n_clips": n_clips, "missing": missing, "complete": not missing,
               "summary": {"q2c": summ}, "records": sorted(recs, key=lambda r: r["index"])}
        json.dump(out, open(os.path.join(args.out_dir, "q2c_manifest.json"), "w"), indent=1)
        print(json.dumps({"n_done": len(recs), "missing": missing[:20], "summary": summ}, indent=1))
        return

    import f5_whitebox_pgd as wb
    from f5_tts.api import F5TTS
    ds = ce.load_dataset(args, crop=args.max_full_seconds)
    backbone = cwe.build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    f5a = wb.F5Attacker(F5TTS(device=device), steps=q2["f5_steps"], cfg=q2["f5_cfg"])
    ecapa = wb.load_ecapa_differentiable(device)
    verifiers = wb.make_verifiers(ecapa, wb.load_xvector(device), wb.load_wavlm_sv_differentiable(device))
    content = CtcContentLoss(device, q2["pgd_text"])
    transcribe = Transcriber(device)

    lo, hi = (50, 51) if args.diagnostic else (args.start, min(args.stop if args.stop is not None else n_clips, n_clips))
    if not args.diagnostic:
        os.makedirs(rec_dir, exist_ok=True)
    t0, n_new = time.time(), 0
    for i in range(lo, hi):
        rp, fp = os.path.join(rec_dir, f"{i}.json"), os.path.join(args.out_dir, "q2c", f"{i}.flac")
        if not args.diagnostic and os.path.exists(rp) and os.path.exists(fp):
            continue
        it = ds[i]
        clean = ce.make_prompt(ce.trim_padding(it["waveform"].reshape(-1)), str(q2["prompt_seconds"]), args.max_full_seconds).to(device).reshape(1, 1, -1)
        assert wav_sha(clean) == base[i]["clean_sha"], f"clip {i}: clean prompt differs from the other arms"
        msg = ce.message_bits(q2["message_seed_base"] + i, device)
        ref_text = base[i]["ref_text_pgd"]                       # the transcript Q2 used for this clip
        _t = time.time()
        _wm, out, _d, diag = wb.whitebox_pgd(
            backbone, f5a, clean, msg, ref_text, q2["pgd_text"], q2["epsilon"], q2["step_size"], q2["n_steps"], "wens",
            lambda_wm=q2["lambda_wm"], dis_to_wm_ratio=q2["dis_to_wm_ratio"], ecapa_model=ecapa,
            noise_seed=q2["noise_seed_base"] + i, f5_seed=q2["f5_seed_base"] + i, verbose=args.diagnostic,
            verifiers=verifiers, shape_alpha=q2["shape_alpha"], shape_floor=q2["shape_floor"],
            content_fn=content, content_weight=args.content_weight)
        sec = time.time() - _t
        if args.diagnostic:
            import tempfile
            fp = os.path.join(tempfile.mkdtemp(), "q2c.flac")
        write_flac(fp, out)
        back = read_audio(fp, device)
        dec = np.array(ce.decode_bits(backbone, back))
        cts = diag.get("ct_loss", [])
        rec = {"index": i, "utterance_id": base[i]["utterance_id"], "clean_sha": base[i]["clean_sha"], "pgd_seconds": round(sec, 1),
               "q2c": {"acc_source": float((dec == np.array(msg[0].tolist())).mean()), **quality(clean, back)},
               "diag": {"skipped": diag["skipped"], "dis_loss_first": diag["dis_loss"][0], "dis_loss_last": diag["dis_loss"][-1],
                        "ct_loss_first": cts[0] if cts else None, "ct_loss_last": cts[-1] if cts else None,
                        "ct_nonfinite_steps": diag.get("ct_nonfinite", 0), "ct_zero_steps": diag.get("ct_zero", 0),
                        "dis_nonfinite_steps": diag.get("dis_nonfinite", 0), "ct_short_rollouts": content.n_short}}
        content.n_short = 0
        if args.diagnostic:
            print(json.dumps(rec, indent=1))
            assert cts, "the content term was not used"
            print(f"steps: skipped {diag['skipped']}, speaker term dropped {diag.get('dis_nonfinite', 0)}, "
                  f"content term non-finite {diag.get('ct_nonfinite', 0)}, content term zero {diag.get('ct_zero', 0)} of {q2['n_steps']}")
            assert diag["skipped"] <= 10, f"{diag['skipped']} of {q2['n_steps']} PGD steps skipped: send the output above to Claude"
            assert diag.get("dis_nonfinite", 0) <= 10, "the speaker gradient was non-finite on more than 10 steps: send the output above to Claude"
            assert diag.get("ct_nonfinite", 0) + diag.get("ct_zero", 0) < q2["n_steps"] // 2, \
                "the content gradient was unusable on most steps: send the output above to Claude"
            if not cts[-1] > cts[0]:
                print("WARNING: the CTC loss of the rollout did not rise on this clip (one clip; see the step log)")
            print("DIAGNOSTIC OK")
            return
        json.dump(rec, open(rp, "w"), indent=1)
        n_new += 1
        el = time.time() - t0
        print(f"[{i}] q2c acc {rec['q2c']['acc_source']:.3f} pesq {rec['q2c']['pesq']:.2f} dis {rec['diag']['dis_loss_first']:.3f}->"
              f"{rec['diag']['dis_loss_last']:.3f} ctc {rec['diag']['ct_loss_first']:.2f}->{rec['diag']['ct_loss_last']:.2f} "
              f"skip={diag['skipped']} | {el / n_new:.0f} s/clip, ~{el / n_new * (hi - i - 1) / 60:.0f} min left", flush=True)
    print(f"DONE clips {lo}..{hi - 1}")


if __name__ == "__main__":
    main()
