"""
src/eval/demucs_fallback_eval.py

THE THESIS EXPERIMENT: attribution as a fallback for the case where adversarial
protection is defeated.

WHY THIS SPECIFIC ATTACK, AND WHY IT IS NOT ANOTHER PURIFICATION TEST
---------------------------------------------------------------------
SafeSpeech's own paper (USENIX Security 2025, Section on adaptive attacks) reports
that strong denoising is what degrades their protection most:

    DEMUCS denoising:  WER 99.610% -> 57.329%,  SIM 0.204 -> 0.284
    AudioPure:         WER 99.610% -> 85.711%

So DEMUCS is roughly FIVE TIMES more damaging to SafeSpeech than AudioPure is. Every
purification experiment in this project so far used AudioPure -- i.e. the attack
SafeSpeech already largely survives. This script targets the attack the authors
themselves identify as their weak point.

The gap this creates is structural, not incremental:

    SafeSpeech under DEMUCS  ->  protection partially stripped, attacker gets a
                                 usable clone, AND NO TRACEABILITY EXISTS, because
                                 SafeSpeech provides no watermark at all.

    This system under DEMUCS ->  protection partially stripped in the same way, BUT
                                 the watermark may still identify the source.

If the watermark survives DEMUCS, the combined system provides recourse in exactly
the scenario where SafeSpeech provides none. That is a measurable improvement over
SafeSpeech on a failure mode SafeSpeech's own authors published -- not one invented
here, and not a re-solving of something a baseline already handles.

WHAT IS MEASURED (per utterance, all in one pass)
--------------------------------------------------
  acc_wm                  watermarked audio, no perturbation, no denoising  [sanity]
  acc_wm_denoised         watermarked -> DEMUCS                             [does denoising
                                                                             alone hurt the mark?]
  acc_protected           watermarked + PGD perturbation                    [control]
  acc_protected_denoised  watermarked + PGD -> DEMUCS   <-- THE KEY NUMBER  [attribution
                                                                             after the attack]
  sim_clean_clone         clone of clean audio                              [no protection]
  sim_protected_clone     clone of protected audio                          [protection working]
  sim_denoised_clone      clone of DEMUCS-denoised protected audio          [protection after attack]

Derived:
  protection_retained  = (sim_clean - sim_denoised) / (sim_clean - sim_protected)
                         1.0 = denoising did nothing; 0.0 = protection fully stripped.
  attribution_retained = (acc_protected_denoised - 0.5) / (acc_protected - 0.5)
                         how much of the recoverable watermark signal survives DEMUCS.

HOW TO READ THE RESULT (decide this BEFORE running -- pre-registered)
----------------------------------------------------------------------
  CONTRIBUTION CONFIRMED if acc_protected_denoised is significantly above chance
  (paired/binomial test, n=100, p < 0.05) while protection_retained shows measurable
  degradation. That is: protection weakens under DEMUCS exactly as SafeSpeech reports,
  but attribution survives -- the fallback SafeSpeech structurally cannot offer.

  CONTRIBUTION NOT SUPPORTED if acc_protected_denoised is at chance. Then DEMUCS
  removes both protection and attribution, there is no fallback, and the thesis falls
  back to the compositionality + characterisation result (already fully evidenced).

Either outcome is a decision, not an invitation to invent a new route.

DENOISER
--------
SafeSpeech says "DEMUCS". For speech enhancement the standard is Facebook's `denoiser`
package (Defossez et al., "Real Time Speech Enhancement in the Waveform Domain"), which
is Demucs-architecture and operates natively at 16 kHz -- this project's sample rate.
Falls back to the `demucs` music-separation package's vocals stem if `denoiser` is
absent. The script PRINTS which backend it used; report that in the thesis, because
they are not the same model.

    pip install denoiser          # preferred
    pip install demucs            # fallback

Usage:
    # 1. ALWAYS first -- verifies the denoiser loads and produces sane audio:
    python src/eval/demucs_fallback_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --diagnostic

    # 2. Quick read (~25 utterances):
    python src/eval/demucs_fallback_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --n_utterances 25 --output results/results_demucs_fallback_n25.json

    # 3. Full run:
    python src/eval/demucs_fallback_eval.py \\
        --checkpoint ./checkpoints/stage1_final_scaleup_recalibrated/recalibrated_final.pt \\
        --n_speakers 60 --n_eval_speakers 20 --eval_utterances_per_speaker 5 \\
        --n_utterances 100 --output results/results_demucs_fallback_n100.json
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(__file__))

from librispeech import LibriSpeechSubset, collate_librispeech
from surrogate_vc import load_yourtts_surrogate
from disruption_pgd import build_backbone, random_message, detect_acc, compute_sim, pgd_perturb


def _fix_length(out, ref):
    """Denoisers can return a slightly different length; align to the input."""
    if out.shape[-1] == ref.shape[-1]:
        return out
    if out.shape[-1] > ref.shape[-1]:
        return out[..., : ref.shape[-1]]
    return torch.nn.functional.pad(out, (0, ref.shape[-1] - out.shape[-1]))


def build_demucs_denoiser(device, backend="auto"):
    """
    Returns (denoise_fn, backend_name). denoise_fn takes [B,1,T] @16kHz and returns
    the same shape.

    DEPENDENCY WARNING (learned the hard way, 2026-09-10): `pip install denoiser`
    pulls in an old omegaconf (<2.0, no `omegaconf.base`), which makes torch.load
    fail on the VoiceMark checkpoint with ModuleNotFoundError. If you install it,
    repair with `pip install "omegaconf>=2.0.6"` afterwards and re-verify the
    checkpoint loads. The `torchaudio` backend below needs NO new package and
    therefore cannot break the environment -- prefer it unless you specifically
    need speech-domain enhancement.

    backend: "auto" | "torchaudio" | "denoiser" | "demucs"
      torchaudio : Hybrid Demucs (HDEMUCS_HIGH_MUSDB_PLUS), music separation,
                   vocals stem used as the denoiser. No install. Cannot break anything.
      denoiser   : Facebook speech enhancement (Demucs arch, 16 kHz native). Closest
                   to what SafeSpeech's paper most likely means, but see the warning.
      demucs     : standalone music-separation package, vocals stem.
    """
    errs = {}

    if backend in ("auto", "torchaudio"):
        try:
            import torchaudio
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model().to(device).eval()
            model_sr = bundle.sample_rate
            v_idx = list(model.sources).index("vocals")
            print(f"[demucs] backend = torchaudio HDEMUCS_HIGH_MUSDB_PLUS, vocals stem "
                  f"(native {model_sr} Hz, resampling 16k<->{model_sr}). No extra install.")

            def _denoise(wav):
                with torch.no_grad():
                    x = torchaudio.functional.resample(wav, 16000, model_sr)
                    if x.shape[1] == 1:
                        x = x.repeat(1, 2, 1)          # HDemucs expects stereo
                    est = model(x)                      # [B, sources, 2, T]
                    voc = est[:, v_idx].mean(dim=1, keepdim=True)
                    out = torchaudio.functional.resample(voc, model_sr, 16000)
                return _fix_length(out, wav)

            return _denoise, "torchaudio.hdemucs_vocals"
        except Exception as e:
            errs["torchaudio"] = f"{type(e).__name__}: {e}"
            if backend == "torchaudio":
                raise RuntimeError(f"torchaudio HDemucs unavailable -- {errs['torchaudio']}")
            print(f"[demucs] torchaudio HDemucs unavailable ({errs['torchaudio']}); trying `denoiser`.")

    if backend in ("auto", "denoiser"):
      try:
        from denoiser.pretrained import dns64
        model = dns64().to(device).eval()
        print("[demucs] backend = facebook `denoiser` dns64 (Demucs arch, 16 kHz speech enhancement)")

        def _denoise(wav):
            with torch.no_grad():
                out = model(wav)
            return _fix_length(out, wav)

        return _denoise, "denoiser.dns64"
      except Exception as e:
        errs["denoiser"] = f"{type(e).__name__}: {e}"
        if backend == "denoiser":
            raise RuntimeError(f"`denoiser` unavailable -- {errs['denoiser']}")
        print(f"[demucs] `denoiser` unavailable ({errs['denoiser']}); "
              f"trying music-separation `demucs` vocals stem instead.")

    try:
        import torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        model = get_model("htdemucs").to(device).eval()
        sources = model.sources
        v_idx = sources.index("vocals")
        model_sr = model.samplerate
        print(f"[demucs] backend = `demucs` htdemucs vocals stem "
              f"(music separation, native {model_sr} Hz -- resampling 16k<->{model_sr})")

        def _denoise(wav):
            with torch.no_grad():
                x = torchaudio.functional.resample(wav, 16000, model_sr)
                x = x.repeat(1, 2, 1) if x.shape[1] == 1 else x  # htdemucs wants stereo
                est = apply_model(model, x, device=wav.device, progress=False)
                voc = est[:, v_idx].mean(dim=1, keepdim=True)
                out = torchaudio.functional.resample(voc, model_sr, 16000)
            return _fix_length(out, wav)

        return _denoise, "demucs.htdemucs_vocals"
    except Exception as e:
        errs["demucs"] = f"{type(e).__name__}: {e}"

    raise RuntimeError(
        "No DEMUCS backend could be loaded. Tried: "
        + "; ".join(f"{k} -> {v}" for k, v in errs.items())
        + ".\nThe torchaudio backend needs NO install and should normally work "
          "(torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS); if it failed, torchaudio is "
          "likely too old. Otherwise: pip install denoiser -- but then REPAIR omegaconf "
          "afterwards (pip install 'omegaconf>=2.0.6') or the VoiceMark checkpoint will "
          "no longer unpickle."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--diagnostic", action="store_true",
                   help="One utterance, verbose. Verifies the denoiser loads and does not "
                        "destroy the audio. ALWAYS run this first.")
    p.add_argument("--n_utterances", type=int, default=100)
    p.add_argument("--save_clones_dir", type=str, default=None,
                   help="Write reference + clone WAVs here so SIM can be RE-SCORED with "
                        "ECAPA-TDNN via ecapa_sim_eval.py. Required for any comparison "
                        "against SafeSpeech: their SIM and their 0.25 threshold are defined "
                        "on ECAPA-TDNN, while this script's SIM uses YourTTS's encoder -- "
                        "cosines are not comparable across embedding spaces.")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "torchaudio", "denoiser", "demucs"],
                   help="Which DEMUCS implementation to attack with. 'torchaudio' needs no "
                        "install and cannot break the environment; 'denoiser' is closest to "
                        "SafeSpeech's likely meaning but its install downgrades omegaconf and "
                        "breaks checkpoint loading (repair: pip install 'omegaconf>=2.0.6'). "
                        "Whichever is used is printed AND saved in the output JSON -- report it.")

    p.add_argument("--epsilon", type=float, default=0.002, help="Established operating point.")
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--step_size", type=float, default=None)
    p.add_argument("--random_start", action="store_true", default=True)
    p.add_argument("--lambda_wm", type=float, default=1.0,
                   help="1.0 = the combined system (watermark preserved during PGD). "
                        "0.0 = SafeSpeech-style arm, no watermark-preservation term.")

    p.add_argument("--surrogate_text", type=str, default="This is a test sentence for voice cloning.")
    p.add_argument("--data_root", type=str, default="./data/librispeech")
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--include_ffn", action="store_true")
    p.add_argument("--capacity_lora_r", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step_size = args.step_size if args.step_size is not None else args.epsilon / 4

    eval_ds = LibriSpeechSubset(
        root=args.data_root, n_speakers=args.n_speakers,
        utterances_per_speaker=args.utterances_per_speaker,
        n_eval_speakers=args.n_eval_speakers,
        eval_utterances_per_speaker=args.eval_utterances_per_speaker,
        sample_rate=16000, crop_seconds=args.crop_seconds, split="eval",
    )
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_librispeech)

    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,
                              args.include_ffn, args.capacity_lora_r)
    backbone.model.eval()
    print("[main] Loading YourTTS surrogate (cloner + SIM embedding)...")
    surrogate = load_yourtts_surrogate(device=device)
    denoise, backend = build_demucs_denoiser(device, backend=args.backend)

    # ---------------- diagnostic ----------------
    if args.diagnostic:
        print("\n" + "=" * 64 + "\nDIAGNOSTIC MODE\n" + "=" * 64)
        batch = next(iter(eval_loader))
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123)

        recon_wm, perturbed, _d = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
        )
        den = denoise(perturbed.detach())

        print(f"\n--- shapes / sanity ---")
        print(f"  clean={tuple(clean_audio.shape)} protected={tuple(perturbed.shape)} "
              f"denoised={tuple(den.shape)}")
        print(f"  RMS  clean={clean_audio.pow(2).mean().sqrt():.5f}  "
              f"protected={perturbed.pow(2).mean().sqrt():.5f}  denoised={den.pow(2).mean().sqrt():.5f}")
        print(f"  -> denoised RMS collapsing to ~0 would mean the denoiser destroyed the audio; "
              f"a similar RMS means it behaved sensibly.")

        print(f"\n--- watermark ---")
        a_wm = detect_acc(backbone, recon_wm, message)
        a_prot = detect_acc(backbone, perturbed, message)
        a_den = detect_acc(backbone, den, message)
        print(f"  ACC watermarked (no perturbation):        {a_wm:.4f}")
        print(f"  ACC protected (watermark + PGD):          {a_prot:.4f}")
        print(f"  ACC protected -> DEMUCS  [KEY]:           {a_den:.4f}")
        print(f"  -> at n=1 these are indicative only; 0.5 is chance.")
        print(f"\n[main] Diagnostic complete (backend={backend}). If the audio survived and ACC "
              f"before denoising looks right, drop --diagnostic and run for real.")
        return

    # ---------------- full run ----------------
    keys = ["acc_wm", "acc_wm_denoised", "acc_protected", "acc_protected_denoised",
            "sim_clean_clone", "sim_protected_clone", "sim_denoised_clone",
            # ADDED 2026-09-10 -- the metrics the research question actually needs.
            # Everything above detects the watermark on the AUDIO. These detect it on
            # the CLONE, i.e. "can the deepfake be traced back", which is a different
            # and harder question than "can this leaked file be proven mine".
            "acc_clone_clean", "acc_clone_protected", "acc_clone_denoised"]
    m = {k: [] for k in keys}

    print(f"\n{'=' * 78}")
    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "
          f"lambda_wm={args.lambda_wm} | n={args.n_utterances}")
    print(f"SafeSpeech's own DEMUCS result for reference: WER 99.6%->57.3%, SIM 0.204->0.284")
    print(f"{'=' * 78}")

    for i, batch in enumerate(eval_loader):
        if i >= args.n_utterances:
            break
        clean_audio = batch["waveform"].to(device)
        message = random_message(16, clean_audio.shape[0], device, seed=123 + i)

        recon_wm, perturbed, _d = pgd_perturb(
            backbone, surrogate, clean_audio, message, args.surrogate_text,
            args.epsilon, step_size, args.n_steps, args.random_start, lambda_wm=args.lambda_wm,
        )
        perturbed = perturbed.detach()

        with torch.no_grad():
            wm_den = denoise(recon_wm.detach())
            prot_den = denoise(perturbed)

            m["acc_wm"].append(detect_acc(backbone, recon_wm, message))
            m["acc_wm_denoised"].append(detect_acc(backbone, wm_den, message))
            m["acc_protected"].append(detect_acc(backbone, perturbed, message))
            m["acc_protected_denoised"].append(detect_acc(backbone, prot_den, message))

            c_clean = surrogate.clone_voice(clean_audio, text=args.surrogate_text)
            c_prot = surrogate.clone_voice(perturbed, text=args.surrogate_text)
            c_den = surrogate.clone_voice(prot_den, text=args.surrogate_text)
            m["sim_clean_clone"].append(compute_sim(surrogate, clean_audio, c_clean))
            m["sim_protected_clone"].append(compute_sim(surrogate, clean_audio, c_prot))
            m["sim_denoised_clone"].append(compute_sim(surrogate, clean_audio, c_den))

            # THE RESEARCH-QUESTION METRICS: watermark detected IN THE CLONE, not in
            # the audio. acc_clone_denoised completes the attacker's actual pipeline --
            # protect -> denoise -> clone -> can the clone still be attributed?
            m["acc_clone_clean"].append(detect_acc(backbone, c_clean, message))
            m["acc_clone_protected"].append(detect_acc(backbone, c_prot, message))
            m["acc_clone_denoised"].append(detect_acc(backbone, c_den, message))

            if args.save_clones_dir:
                import soundfile as sf
                os.makedirs(args.save_clones_dir, exist_ok=True)

                def _w(tag, wav, subdir=None):
                    d = os.path.join(args.save_clones_dir, subdir) if subdir else args.save_clones_dir
                    os.makedirs(d, exist_ok=True)
                    sf.write(os.path.join(d, f"sample{i}_{tag}.wav"),
                             wav.detach().cpu().reshape(-1).numpy(), 16000)

                # Top level: reference + clones. ecapa_sim_eval.py / clone_wer_eval.py
                # glob "sample*_*.wav" at THIS level only, so anything in a subdirectory
                # is invisible to them -- which is why the quality pair goes below.
                _w("reference", clean_audio[0])
                _w("clone_clean", c_clean[0])
                _w("clone_protected", c_prot[0])
                _w("clone_demucs", c_den[0])

                # audio/ subdir: the human-side quality pair, named exactly as
                # quality_metrics.py expects (sampleN_clean / sampleN_watermarked), so
                # PESQ/STOI/SI-SNR on the PUBLISHED audio runs with no new code:
                #   python src/eval/quality_metrics.py --sample_dir <dir>/audio \
                #       --n_samples 100 --skip_wer
                _w("clean", clean_audio[0], subdir="audio")
                _w("watermarked", perturbed[0], subdir="audio")
                _w("denoised", prot_den[0], subdir="audio")

        print(f"  [{i}] acc: wm={m['acc_wm'][-1]:.4f} prot={m['acc_protected'][-1]:.4f} "
              f"prot+DEMUCS={m['acc_protected_denoised'][-1]:.4f} | "
              f"sim: clean={m['sim_clean_clone'][-1]:.4f} prot={m['sim_protected_clone'][-1]:.4f} "
              f"prot+DEMUCS={m['sim_denoised_clone'][-1]:.4f}", flush=True)

        # Incremental save every 10 utterances -- a session timeout must not cost the run.
        if args.output and (i + 1) % 10 == 0:
            means = {k: sum(v) / len(v) for k, v in m.items()}
            with open(args.output, "w") as f:
                json.dump({"label": "demucs_fallback", "backend": backend,
                           "checkpoint": args.checkpoint, "n_completed": i + 1,
                           "results": {**means, **{f"{k}_values": v for k, v in m.items()}}},
                          f, indent=2)

    means = {k: sum(v) / len(v) for k, v in m.items()}

    sc, sp, sd = means["sim_clean_clone"], means["sim_protected_clone"], means["sim_denoised_clone"]
    protection_retained = (sc - sd) / (sc - sp) if abs(sc - sp) > 1e-6 else float("nan")
    ap, ad = means["acc_protected"], means["acc_protected_denoised"]
    attribution_retained = (ad - 0.5) / (ap - 0.5) if abs(ap - 0.5) > 1e-6 else float("nan")

    print(f"\n{'=' * 78}")
    print(f"RESULT  (backend={backend}, n={len(m['acc_wm'])})")
    print(f"{'=' * 78}")
    print(f"  ATTRIBUTION -- SCENARIO A: can a LEAKED FILE be proven yours?")
    print(f"    (watermark detected on the AUDIO itself)")
    print(f"    watermarked, no attack:          {means['acc_wm']:.4f}")
    print(f"    watermarked -> DEMUCS:           {means['acc_wm_denoised']:.4f}")
    print(f"    protected (wm + PGD):            {ap:.4f}")
    print(f"    protected -> DEMUCS:             {ad:.4f}")
    print(f"    attribution retained:            {attribution_retained * 100:.1f}%")
    print(f"\n  ATTRIBUTION -- SCENARIO B: can the CLONE be traced back?")
    print(f"    (watermark detected on the CLONE -- this is what the research question asks)")
    print(f"    clone of clean audio:            {means['acc_clone_clean']:.4f}")
    print(f"    clone of protected audio:        {means['acc_clone_protected']:.4f}")
    print(f"    clone after DEMUCS    [KEY]:     {means['acc_clone_denoised']:.4f}")
    print(f"    -> 0.5 is chance. This completes the attacker's real pipeline:")
    print(f"       protect -> denoise -> clone -> attribute. Scenario A does NOT imply B.")
    print(f"\n  PROTECTION")
    print(f"    SIM, clone of clean audio:       {sc:.4f}")
    print(f"    SIM, clone of protected audio:   {sp:.4f}")
    print(f"    SIM, clone after DEMUCS:         {sd:.4f}")
    print(f"    protection retained:             {protection_retained * 100:.1f}%")
    print(f"\n  READ:")
    print(f"    If protection degraded (as SafeSpeech reports under DEMUCS) but the KEY")
    print(f"    attribution number is significantly above 0.5, the combined system provides")
    print(f"    recourse exactly where SafeSpeech provides none. Run the significance test")
    print(f"    on acc_protected_denoised_values before claiming it.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "label": "demucs_fallback", "backend": backend, "checkpoint": args.checkpoint,
                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,
                        "lambda_wm": args.lambda_wm, "step_size": step_size},
                "n_completed": len(m["acc_wm"]),
                "safespeech_reference": {"wer_before": 0.996, "wer_after_demucs": 0.573,
                                          "sim_before": 0.204, "sim_after_demucs": 0.284},
                "results": {**means,
                            "protection_retained": protection_retained,
                            "attribution_retained": attribution_retained,
                            **{f"{k}_values": v for k, v in m.items()}},
            }, f, indent=2)
        print(f"\n[main] Saved to {args.output}")


if __name__ == "__main__":
    main()
