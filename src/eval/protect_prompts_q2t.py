"""
src/eval/protect_prompts_q2t.py  (2026-10-26, pre-registered in docs/preregistration_q2t_2026-10-26.md, part B)

Q2 plus a prompt-transcription attack, after E2E-VGuard (Section 3.4: a targeted attack on the ASR that transcribes the
prompt, so the cloner is given a transcript that does not match the prompt audio). Two arms, both on the clips, crop,
messages, bound and seeds of protect_prompts_v2.py (its manifest is reused and every clean prompt is hash-checked):

    q2t : Q2 exactly as registered (F5-TTS surrogate rollout, three verifiers, shaped bound peak 0.015, 40 steps,
          step 0.0015, dis:wm 1.732, lambda_wm 1, seeds 5000 + i / 7000 + i)  PLUS  a targeted ASR term on the prompt
    t   : the ASR term and the watermark term only (no speaker term, no F5 rollout), same bound, steps and start seed:
          says how much of q2t comes from the transcription attack alone

ASR term. Surrogate recogniser: openai/whisper-base.en (Hugging Face weights, differentiable log-mel written here and
checked against WhisperFeatureExtractor in --diagnostic). Loss = teacher-forced cross-entropy of a TARGET transcript,
minimised (targeted, as E2E-VGuard). Target for clip i = the first N words of the LibriSpeech transcript of clip
(i + 37) mod 100 (always another speaker), N = the number of words faster-whisper base.en hears in the clean prompt
(E2E-VGuard's "text of the same length"), at least 3. Its gradient is rescaled to --prompt_weight x ||grad_wm||
(default 1.732, the weight of the speaker term).

Which recogniser the cloners use (so what is matched and what is held out):
    MaskGCT, CosyVoice : faster-whisper base.en in clone_eval_v2 (CTranslate2 conversion of the SAME weights as the
                         surrogate: matched); sensitivity run with large-v3-turbo (held out)
    F5-TTS             : its own transcription (ref_text ""), Whisper large-v3-turbo: held out

Output (out_dir = the protect_prompts_v2.py dir of the checkpoint): ARM/{i}.flac (16-bit), records_ARM/{i}.json,
--merge writes ARM_manifest.json.

    python src/eval/protect_prompts_q2t.py --arm t   --checkpoint CKPT --out_dir results/anticlone/prompts/RQA DATA --start 0 --stop 100
    python src/eval/protect_prompts_q2t.py --arm q2t ... ;  ... --merge ;  ... --diagnostic (clip 50, verbose, writes nothing)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "models", "data", "losses"):
    sys.path.insert(0, os.path.join(HERE, "..", sub) if sub else os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import clone_eval_v2 as ce  # noqa: E402
import cloner_watermark_eval as cwe  # noqa: E402
from protect_prompts_v2 import Transcriber, quality, read_audio, wav_sha, write_flac  # noqa: E402

TARGET_OFFSET = 37
N_SAMPLES = 480000                     # Whisper's 30 s window at 16 kHz


class WhisperTargetLoss:
    """Teacher-forced CE of a target transcript under openai/whisper-base.en, differentiable in the waveform."""

    def __init__(self, device, name="openai/whisper-base.en"):
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        self.tok = WhisperTokenizer.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
        for p_ in self.model.parameters():
            p_.requires_grad_(False)
        self.device = device
        self.mel = torch.tensor(np.asarray(self.fe.mel_filters), dtype=torch.float32, device=device)   # [201, 80]
        self.win = torch.hann_window(self.fe.n_fft, periodic=True, device=device)
        self.sot = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.nots = self.tok.convert_tokens_to_ids("<|notimestamps|>")
        self.eot = self.tok.eos_token_id
        self.ids = None

    def features(self, wave16k):
        """Same steps as WhisperFeatureExtractor: pad to 30 s, |STFT|^2 (400, hop 160, periodic Hann, centred, reflect),
        mel, log10 (floor 1e-10), drop last frame, max - 8 floor, (x + 4) / 4."""
        x = wave16k.reshape(-1).float()
        x = F.pad(x, (0, max(0, N_SAMPLES - x.shape[-1])))[:N_SAMPLES]
        st = torch.stft(x, self.fe.n_fft, self.fe.hop_length, window=self.win, center=True, pad_mode="reflect", return_complex=True)
        spec = st.abs() ** 2                                                      # [201, frames]
        logm = torch.log10(torch.clamp(self.mel.T @ spec, min=1e-10))[:, :-1]
        logm = torch.maximum(logm, logm.max() - 8.0)
        return ((logm + 4.0) / 4.0).unsqueeze(0)                                  # [1, 80, 3000]

    def set_target(self, text):
        t = self.tok.encode(" " + text.strip(), add_special_tokens=False)
        self.ids = torch.tensor([[self.sot, self.nots] + t + [self.eot]], device=self.device)

    def __call__(self, wave16k):
        logits = self.model(input_features=self.features(wave16k), decoder_input_ids=self.ids[:, :-1]).logits
        return F.cross_entropy(logits[0, 1:].float(), self.ids[0, 2:])              # predictions from <|notimestamps|> on

    @torch.no_grad()
    def decode(self, wave16k):
        out = self.model.generate(self.features(wave16k), max_new_tokens=60)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def check_features(self, wave16k):
        ref = self.fe(wave16k.reshape(-1).detach().cpu().numpy().astype(np.float32), sampling_rate=16000, return_tensors="np").input_features
        with torch.no_grad():
            mine = self.features(wave16k).cpu().numpy()
        return float(np.abs(ref - mine).max())


def target_text(ds_items, i, n_words):
    j = (i + TARGET_OFFSET) % len(ds_items)
    assert ds_items[j]["speaker"] != ds_items[i]["speaker"], (i, j)
    w = ds_items[j]["transcript"].lower().split()
    return ce.normalize_text(" ".join(w[:max(3, n_words)])), j


def asr_only_pgd(backbone, clean, message, loss_fn, epsilon, step_size, n_steps, shape_alpha, shape_floor, lambda_wm,
                 prompt_weight, noise_seed, verbose=False):
    """arm t: whitebox_pgd's update with the speaker term removed (no F5 rollout)."""
    import f5_whitebox_pgd as wb
    from voicemark_losses import compute_ldec
    with torch.no_grad():
        recon_wm = backbone.forward_full(clean, message)["recon_wm"].detach()
    bound = wb.envelope_bound(recon_wm, epsilon, shape_alpha, shape_floor)
    step_vec = step_size * (bound / epsilon)
    delta = wb._random_start(recon_wm, bound, noise_seed).detach().requires_grad_(True)
    diag = {"pt_loss": [], "wm_loss": [], "skipped": 0, "mean_bound": float(bound.mean().item())}
    for step in range(n_steps):
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)
        pt_loss = loss_fn(perturbed)
        g_pt = torch.autograd.grad(pt_loss, delta)[0]
        perturbed = torch.clamp(recon_wm + delta, -1.0, 1.0)
        with torch.backends.cudnn.flags(enabled=False):
            feat = backbone.model.st_model.forward_feature(perturbed)
            _l, chunk_logits = backbone.model.detector(feat)
            wm_loss = compute_ldec(chunk_logits, message)
        g_wm = torch.autograd.grad(wm_loss, delta)[0]
        gn_wm, gn_pt = g_wm.norm().item(), g_pt.norm().item()
        grad = lambda_wm * g_wm + prompt_weight * (gn_wm / max(gn_pt, 1e-12)) * g_pt
        diag["pt_loss"].append(float(pt_loss.item())); diag["wm_loss"].append(float(wm_loss.item()))
        if not torch.isfinite(grad).all():
            diag["skipped"] += 1
            continue
        if verbose:
            print(f"  [t step {step}] asr_ce={pt_loss.item():.3f} wm_loss={wm_loss.item():.4f} |g_pt|={gn_pt:.3e} |g_wm|={gn_wm:.3e}")
        with torch.no_grad():
            delta = torch.max(torch.min(delta - step_vec * grad.sign(), bound), -bound)
        delta = delta.detach().requires_grad_(True)
    with torch.no_grad():
        out = torch.clamp(recon_wm + delta, -1.0, 1.0)
    return recon_wm, out, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("q2t", "t"), required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    p.add_argument("--prompt_weight", type=float, default=1.732)
    p.add_argument("--asr_model", default="openai/whisper-base.en")
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
    A = args.arm

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.load(open(os.path.join(args.out_dir, "manifest.json")))
    assert man["complete"], "run protect_prompts_v2.py (and its --merge) for this checkpoint first"
    ck_sha = ce.sha256(args.checkpoint)
    assert man["settings"]["checkpoint_sha256"] == ck_sha, "--checkpoint differs from the one the other arms were made with"
    q2 = man["settings"]
    base = {r["index"]: r for r in man["records"]}
    n_clips = min(args.n_utterances, man["n_clips"])
    rec_dir = os.path.join(args.out_dir, f"records_{A}")

    if args.merge:
        recs = [json.load(open(os.path.join(rec_dir, f"{i}.json"))) for i in range(n_clips)
                if os.path.exists(os.path.join(rec_dir, f"{i}.json"))]
        missing = sorted(set(range(n_clips)) - {r["index"] for r in recs})
        summ = {k: float(np.mean([r[A][k] for r in recs])) for k in ("acc_source", "pesq", "stoi", "si_snr", "snr_db")} if recs else {}
        if recs:
            nz = lambda v: np.nan if v is None else v
            summ["prompt_wer_vs_clean_transcript"] = float(np.nanmean([nz(r["asr"]["wer_vs_clean"]) for r in recs]))
            summ["prompt_wer_vs_target"] = float(np.nanmean([nz(r["asr"]["wer_vs_target"]) for r in recs]))
            summ["clips_over_10_skipped"] = [r["index"] for r in recs if r["diag"]["skipped"] > 10]
        out = {"script": "protect_prompts_q2t", "arm": A,
               "settings": {**{k: v for k, v in vars(args).items() if k not in ("start", "stop", "merge", "diagnostic")},
                            "target_offset": TARGET_OFFSET,
                            "q2_settings": {k: q2[k] for k in ("epsilon", "shape_alpha", "shape_floor", "n_steps", "step_size",
                                                                "dis_to_wm_ratio", "lambda_wm", "f5_steps", "f5_cfg", "pgd_text")}},
               "checkpoint_sha256": ck_sha, "n_clips": n_clips, "missing": missing, "complete": not missing,
               "summary": {A: summ}, "records": sorted(recs, key=lambda r: r["index"])}
        json.dump(out, open(os.path.join(args.out_dir, f"{A}_manifest.json"), "w"), indent=1)
        print(json.dumps({"n_done": len(recs), "missing": missing[:20], "summary": summ}, indent=1))
        return

    ds = ce.load_dataset(args, crop=args.max_full_seconds)
    items = [{"speaker": str(ds[j]["speaker_id"]), "transcript": ds[j].get("transcript", "")} for j in range(n_clips)]
    backbone = cwe.build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)
    asr_loss = WhisperTargetLoss(device, args.asr_model)
    transcribe = Transcriber(device)                    # the evaluation's prompt recogniser (faster-whisper base.en)
    if A == "q2t":
        import f5_whitebox_pgd as wb
        from f5_tts.api import F5TTS
        f5a = wb.F5Attacker(F5TTS(device=device), steps=q2["f5_steps"], cfg=q2["f5_cfg"])
        ecapa = wb.load_ecapa_differentiable(device)
        verifiers = wb.make_verifiers(ecapa, wb.load_xvector(device), wb.load_wavlm_sv_differentiable(device))

    lo, hi = (50, 51) if args.diagnostic else (args.start, min(args.stop if args.stop is not None else n_clips, n_clips))
    if not args.diagnostic:
        os.makedirs(rec_dir, exist_ok=True)
    t0, n_new = time.time(), 0
    for i in range(lo, hi):
        rp, fp = os.path.join(rec_dir, f"{i}.json"), os.path.join(args.out_dir, A, f"{i}.flac")
        if not args.diagnostic and os.path.exists(rp) and os.path.exists(fp):
            continue
        it = ds[i]
        clean = ce.make_prompt(ce.trim_padding(it["waveform"].reshape(-1)), str(q2["prompt_seconds"]), args.max_full_seconds).to(device).reshape(1, 1, -1)
        assert wav_sha(clean) == base[i]["clean_sha"], f"clip {i}: clean prompt differs from the other arms"
        msg = ce.message_bits(q2["message_seed_base"] + i, device)
        ref_text = base[i]["ref_text_pgd"]                                    # base.en transcript of the clean prompt
        tgt, j = target_text(items, i, len(ce.words(ref_text)))
        asr_loss.set_target(tgt)
        if args.diagnostic:
            err = asr_loss.check_features(clean)
            print(f"log-mel vs WhisperFeatureExtractor: max |diff| {err:.2e}")
            assert err < 1e-3, "differentiable Whisper features do not match the feature extractor: send this to Claude"
            print("surrogate hears the clean prompt as:", repr(asr_loss.decode(clean)), "| base.en:", repr(ref_text))
            print("target:", repr(tgt), f"(from clip {j})")
        _t = time.time()
        if A == "q2t":
            _wm, out, _d, diag = wb.whitebox_pgd(
                backbone, f5a, clean, msg, ref_text, q2["pgd_text"], q2["epsilon"], q2["step_size"], q2["n_steps"], "wens",
                lambda_wm=q2["lambda_wm"], dis_to_wm_ratio=q2["dis_to_wm_ratio"], ecapa_model=ecapa,
                noise_seed=q2["noise_seed_base"] + i, f5_seed=q2["f5_seed_base"] + i, verbose=args.diagnostic,
                verifiers=verifiers, shape_alpha=q2["shape_alpha"], shape_floor=q2["shape_floor"],
                prompt_fn=asr_loss, prompt_weight=args.prompt_weight)
        else:
            _wm, out, diag = asr_only_pgd(backbone, clean, msg, asr_loss, q2["epsilon"], q2["step_size"], q2["n_steps"],
                                          q2["shape_alpha"], q2["shape_floor"], q2["lambda_wm"], args.prompt_weight,
                                          q2["noise_seed_base"] + i, verbose=args.diagnostic)
        sec = time.time() - _t
        if args.diagnostic:
            import tempfile
            fp = os.path.join(tempfile.mkdtemp(), f"{A}.flac")
        write_flac(fp, out)
        back = read_audio(fp, device)
        dec = np.array(ce.decode_bits(backbone, back))
        heard = transcribe(back)
        pts = diag.get("pt_loss", [])
        rec = {"index": i, "utterance_id": base[i]["utterance_id"], "clean_sha": base[i]["clean_sha"], "pgd_seconds": round(sec, 1),
               A: {"acc_source": float((dec == np.array(msg[0].tolist())).mean()), **quality(clean, back)},
               "asr": {"clean_transcript": ref_text, "target": tgt, "target_from_clip": j, "heard_base_en": heard,
                       "heard_surrogate": asr_loss.decode(back),
                       "wer_vs_clean": ce.wer(ref_text, heard), "wer_vs_target": ce.wer(tgt, heard)},
               "diag": {"skipped": diag["skipped"], "pt_loss_first": pts[0] if pts else None, "pt_loss_last": pts[-1] if pts else None,
                        "pt_nonfinite_steps": diag.get("pt_nonfinite", 0), "dis_nonfinite_steps": diag.get("dis_nonfinite", 0),
                        "dis_loss_first": diag["dis_loss"][0] if diag.get("dis_loss") else None,
                        "dis_loss_last": diag["dis_loss"][-1] if diag.get("dis_loss") else None}}
        if args.diagnostic:
            print(json.dumps(rec, indent=1))
            assert pts, "the ASR term was not used"
            assert diag["skipped"] <= 10, f"{diag['skipped']} of {q2['n_steps']} steps skipped: send the output above to Claude"
            assert diag.get("pt_nonfinite", 0) <= 10, "the ASR gradient was non-finite on more than 10 steps: send the output above to Claude"
            if not pts[-1] < pts[0]:
                print("WARNING: the target cross-entropy did not fall on this clip (one clip; see the step log)")
            print(f"{A}: {sec:.0f} s for this clip")
            print("DIAGNOSTIC OK")
            return
        json.dump(rec, open(rp, "w"), indent=1)
        n_new += 1
        el = time.time() - t0
        print(f"[{i}] {A} acc {rec[A]['acc_source']:.3f} pesq {rec[A]['pesq']:.2f} asr_ce {rec['diag']['pt_loss_first']:.2f}->"
              f"{rec['diag']['pt_loss_last']:.2f} heard {heard[:50]!r} wer_clean {rec['asr']['wer_vs_clean']:.2f} "
              f"| {el / n_new:.0f} s/clip, ~{el / n_new * (hi - i - 1) / 60:.0f} min left", flush=True)
    print(f"DONE clips {lo}..{hi - 1}")


if __name__ == "__main__":
    main()
