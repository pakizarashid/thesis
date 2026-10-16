"""
src/eval/gen_whitebox_audio.py  (H-WHITEBOX, 2026-10-14)

Same job and same output layout as composability_gen_protected_audio.py, but the disruption term is F5-TTS itself (f5_whitebox_pgd.py).
Writes per clip:  sample{i}_reference.wav / sample{i}_unprotected.wav / sample{i}_protected.wav  + _manifest.json + _transcripts.json
(file names are exactly what cloner_watermark_eval.py / ecapa_sim_eval.py / quality_metrics.py already read).

Pairing with the A0 control: clip i uses the PGD random start  torch.Generator.manual_seed(noise_seed_base + i)  -- the same recipe the
H-ENSEMBLE generator wrapper uses -- and the message seed 123 + i, so the white-box arms and A0 start from the same delta0 on every clip.
The F5 rollout noise inside the loss uses f5_seed_base + i; the F5 clones that are EVALUATED draw their own random seeds, so transfer across
sampling noise is part of the test.

Modes:
  --calibrate N : measure ||grad_sim||/||grad_wm|| of the existing YourTTS objective on the first N clips, write whitebox_ratio.json, stop.
  --diagnostic  : 1 clip, verbose, writes nothing.
"""
import os, sys, json, time, argparse, tempfile
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from disruption_pgd import build_backbone, random_message, detect_acc
from librispeech import LibriSpeechSubset, collate_librispeech
import f5_whitebox_pgd as wb


def _write(save_dir, tag, wav):
    os.makedirs(save_dir, exist_ok=True)
    sf.write(os.path.join(save_dir, f"{tag}.wav"), wav.detach().cpu().reshape(-1).numpy(), 16000)


class Transcriber:
    """faster-whisper base.en, the same source cloner_watermark_eval.py uses for --auto_transcribe. Transcribes the CLEAN original once per clip."""
    def __init__(self, device):
        self.device, self._m = device, None

    def __call__(self, wav_np):
        if self._m is None:
            from faster_whisper import WhisperModel
            self._m = WhisperModel("base.en", device=self.device, compute_type="float16" if self.device == "cuda" else "int8")
        # Pass the float32 16 kHz array directly: faster-whisper then skips its file decoder (PyAV), which is version-mismatched on this Kaggle image
        # (TypeError: open() got an unexpected keyword argument 'metadata_errors', 2026-10-14).
        segs, _ = self._m.transcribe(np.asarray(wav_np, dtype=np.float32).reshape(-1), language="en", beam_size=5)
        text = " ".join(s.text for s in segs).strip()
        return text if text else "the quick brown fox jumps over the lazy dog"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--msgproc_lora_r", type=int, default=None)
    p.add_argument("--arm", type=str, choices=["wmel", "wspk", "wens"], default="wspk")
    p.add_argument("--epsilon", type=float, default=0.002)
    p.add_argument("--shape_alpha", type=float, default=0.0, help="0 = flat L-inf ball (default, unchanged behaviour); 1 = bound follows the signal envelope")
    p.add_argument("--shape_floor", type=float, default=0.03, help="lowest bound as a fraction of --epsilon (peak bound) when --shape_alpha > 0")
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--dis_to_wm_ratio", type=float, default=1.0)
    p.add_argument("--f5_steps", type=int, default=6)
    p.add_argument("--f5_cfg", type=float, default=2.0)
    p.add_argument("--gen_text", type=str, default="This is a test sentence for voice cloning.",
                   help="Must equal cloner_watermark_eval.py's --gen_text (its default is the same sentence).")
    p.add_argument("--noise_seed_base", type=int, default=5000)
    p.add_argument("--f5_seed_base", type=int, default=7000)
    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--calibrate", type=int, default=0)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--indices", type=str, default=None, help="comma-separated clip indices to generate (e.g. 0,5,10); overrides --start/--n_utterances; files keep their absolute index")
    p.add_argument("--start", type=int, default=0, help="first clip index to generate (selection run uses 0..19)")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    args = p.parse_args()
    if not (args.diagnostic or args.calibrate) and args.save_dir is None:
        p.error("--save_dir is required unless --diagnostic or --calibrate")

    step_size = args.step_size if args.step_size is not None else args.epsilon / 4.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LibriSpeechSubset(root=args.data_root, n_speakers=args.n_speakers, utterances_per_speaker=args.utterances_per_speaker,
                           n_eval_speakers=args.n_eval_speakers, eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                           sample_rate=16000, crop_seconds=args.crop_seconds, split="eval")
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech, drop_last=False)
    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, include_ffn=False, capacity_lora_r=32, msgproc_lora_r=args.msgproc_lora_r)
    backbone.model.to(device)

    if args.calibrate:
        from surrogate_vc import load_yourtts_surrogate
        surrogate = load_yourtts_surrogate(device=device)
        ratios = []
        for i, batch in enumerate(loader):
            if i >= args.calibrate:
                break
            clean = batch["waveform"].to(device)
            msg = random_message(16, clean.shape[0], device, seed=123 + i)
            gs, gw = wb.reference_grad_norms(backbone, surrogate, clean, msg, args.surrogate_text, args.epsilon, args.noise_seed_base + i)
            ratios.append(gs / max(gw, 1e-12))
            print(f"  [calib {i}] |grad_sim|={gs:.4e} |grad_wm|={gw:.4e} ratio sim:wm = {ratios[-1]:.3f}", flush=True)
        r = float(np.median(ratios))
        json.dump({"median_ratio_sim_to_wm": r, "ratios": ratios, "n": len(ratios)}, open("whitebox_ratio.json", "w"), indent=2)
        print(f"\nCALIBRATION: median ||grad_sim|| / ||grad_wm|| = {r:.3f}  -> use --dis_to_wm_ratio {r:.3f}   (saved whitebox_ratio.json)")
        return

    from f5_tts.api import F5TTS
    print("[main] loading F5-TTS as the differentiable attacker...")
    f5a = wb.F5Attacker(F5TTS(device=device), steps=args.f5_steps, cfg=args.f5_cfg)
    ecapa = wb.load_ecapa_differentiable(device) if args.arm in ("wspk", "wens") else None
    verifiers = None
    if args.arm == "wens":
        verifiers = wb.make_verifiers(ecapa, wb.load_xvector(device), wb.load_wavlm_sv_differentiable(device))
        print("[main] in-loop verifiers:", [n for n, _f in verifiers])
    transcribe = Transcriber(device)
    tpath = os.path.join(args.save_dir, "_transcripts.json") if args.save_dir else None
    transcripts = json.load(open(tpath)) if tpath and os.path.exists(tpath) else {}

    accs_u, accs_p, done, t0 = [], [], [], time.time()
    sel = sorted({int(x) for x in args.indices.split(",")}) if args.indices else None
    first_i, total = (sel[0], len(sel)) if sel else (args.start, args.n_utterances)
    for i, batch in enumerate(loader):
        if sel is not None:
            if i > sel[-1]:
                break
            if i not in sel:
                continue
        else:
            if i < args.start:
                continue
            if i >= args.start + args.n_utterances:
                break
        clean = batch["waveform"].to(device)
        msg = random_message(16, clean.shape[0], device, seed=123 + i)
        ref_text = transcripts.get(str(i)) or transcribe(clean.detach().cpu().reshape(-1).numpy())
        transcripts[str(i)] = ref_text
        _t = time.time()
        recon_wm, perturbed, _d, diag = wb.whitebox_pgd(
            backbone, f5a, clean, msg, ref_text, args.gen_text, args.epsilon, step_size, args.n_steps, args.arm,
            lambda_wm=args.lambda_wm, dis_to_wm_ratio=args.dis_to_wm_ratio, ecapa_model=ecapa,
            noise_seed=args.noise_seed_base + i, f5_seed=args.f5_seed_base + i, verbose=args.diagnostic, verifiers=verifiers,
            shape_alpha=args.shape_alpha, shape_floor=args.shape_floor)
        acc_u, acc_p = detect_acc(backbone, recon_wm, msg), detect_acc(backbone, perturbed, msg)
        if args.diagnostic:
            print(f"\nPGD time: {time.time() - _t:.1f} s for {args.n_steps} steps  | peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            print(f"\ntranscript: {ref_text!r}\nACC unprotected {acc_u:.4f}  protected {acc_p:.4f}\n"
                  f"dis_loss first/last: {diag['dis_loss'][0]:.4f} / {diag['dis_loss'][-1]:.4f}   skipped steps: {diag['skipped']}\n"
                  f"mean bound {diag['mean_bound']:.6f} (peak {args.epsilon}, shape_alpha {args.shape_alpha})")
            return
        accs_u.append(acc_u); accs_p.append(acc_p); done.append(i)
        _write(args.save_dir, f"sample{i}_reference", clean[0]); _write(args.save_dir, f"sample{i}_unprotected", recon_wm[0])
        _write(args.save_dir, f"sample{i}_protected", perturbed[0])
        if i == first_i and acc_u < 0.85:
            raise SystemExit(f"ABORT: acc_unprotected={acc_u:.4f} (expected ~0.99). Check --checkpoint / --msgproc_lora_r.")
        el = time.time() - t0
        print(f"  [{i}] acc_u={acc_u:.4f} acc_p={acc_p:.4f} dis {diag['dis_loss'][0]:.3f}->{diag['dis_loss'][-1]:.3f} skip={diag['skipped']} "
              f"| {el / len(done):.1f} s/clip, ~{el / len(done) * (total - len(done)) / 60:.0f} min left", flush=True)
        if len(done) % 5 == 0:
            json.dump(transcripts, open(tpath, "w"), indent=1)

    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(transcripts, open(tpath, "w"), indent=1)
    manifest = {"label": "hwhitebox_audio_gen", "arm": args.arm, "checkpoint": args.checkpoint, "msgproc_lora_r": args.msgproc_lora_r,
                "pgd": {"epsilon": args.epsilon, "step_size": step_size, "n_steps": args.n_steps, "lambda_wm": args.lambda_wm,
                        "dis_to_wm_ratio": args.dis_to_wm_ratio, "shape_alpha": args.shape_alpha, "shape_floor": args.shape_floor, "f5_steps": args.f5_steps, "f5_cfg": args.f5_cfg,
                        "noise_seed_base": args.noise_seed_base, "f5_seed_base": args.f5_seed_base, "gen_text": args.gen_text},
                "completed_indices": done, "n_completed": len(done),
                "sanity_acc_unprotected_mean": float(np.mean(accs_u)), "sanity_acc_protected_mean": float(np.mean(accs_p))}
    json.dump(manifest, open(os.path.join(args.save_dir, "_manifest.json"), "w"), indent=2)
    print(f"DONE n={len(done)} -> {args.save_dir}  ACC unprotected {manifest['sanity_acc_unprotected_mean']:.4f} protected {manifest['sanity_acc_protected_mean']:.4f}")


if __name__ == "__main__":
    main()
