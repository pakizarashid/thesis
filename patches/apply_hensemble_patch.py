"""
apply_hensemble_patch.py -- standalone catch-up patch, H-ENSEMBLE diagnostic only.

Applies ONLY the H-ENSEMBLE edits (load_ecapa_differentiable/ecapa_embed helpers,
pgd_perturb's ecapa_model/lambda_ecapa_clone/lambda_ecapa_direct/ecapa_native_sr
params, --lambda_ecapa_clone/--lambda_ecapa_direct/--test_ecapa CLI flags, and the
--diagnostic call-site threading) to a disruption_pgd.py that ALREADY has H-SPEC +
H-DIRECT applied -- i.e. your live session's file, as of the H-DIRECT delivery.

Run this in a fresh cell in your Kaggle session:
    %run apply_hensemble_patch.py
then reload the module before the diagnostic cell (same reload guard pattern as
H-SPEC/H-DIRECT):
    import sys, importlib
    if "disruption_pgd" in sys.modules:
        importlib.reload(sys.modules["disruption_pgd"])
        print("disruption_pgd reloaded from disk")
"""
import ast
import os
import sys
import py_compile

# Default matches where every prior patch in this project (H-SPEC/H-DIRECT) targets it:
# src/eval/disruption_pgd.py relative to the repo root (/kaggle/working/thesis in your
# session). Override with a path as the first CLI arg if your checkout differs, e.g.:
#   %run apply_hensemble_patch.py path/to/disruption_pgd.py
_DEFAULT_TARGET = "src/eval/disruption_pgd.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    # Fallback: running from inside src/eval/ already (cwd change), not the repo root.
    TARGET = os.path.basename(TARGET)

EDITS = [
    ('def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,\n                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,\n                 lambda_kl_direct: float = 0.0, lambda_l1n_direct: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n', 'def load_ecapa_differentiable(device):\n    """\n    H-ENSEMBLE (2026-09-12 pre-registration). ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb),\n    loaded for use INSIDE the PGD training loop -- unlike ecapa_sim_eval.py\'s use of the same\n    checkpoint, which wraps every call in torch.no_grad() by design (a separate, dependency-\n    isolated, offline eval process). Here gradients must flow through it, so no_grad is never\n    applied around encode_batch.\n\n    ENVIRONMENT-RISK CANARY: this project has already been bitten once by a similarly-scoped\n    pip install (denoiser silently downgraded omegaconf below 2.0 and broke unpickling the\n    VoiceMark checkpoint -- see ecapa_sim_eval.py\'s own docstring). build_backbone() already\n    runs BEFORE this function is ever called in main() below, so if `pip install speechbrain`\n    damaged the environment the same way, build_backbone\'s checkpoint load fails first, with a\n    clear traceback, before any GPU time is spent on ECAPA -- do not skip straight to\n    --test_ecapa without first confirming build_backbone succeeded in this same process.\n    """\n    from speechbrain.inference.speaker import EncoderClassifier\n    model = EncoderClassifier.from_hparams(\n        source="speechbrain/spkrec-ecapa-voxceleb",\n        savedir="pretrained_models/spkrec-ecapa-voxceleb",\n        run_opts={"device": device},\n    )\n    return model\n\n\ndef ecapa_embed(model, audio: torch.Tensor, native_sr: int):\n    """\n    Differentiable ECAPA-TDNN embedding of an in-memory waveform batch (shape [B, T] or\n    [B, 1, T]), gradient-carrying all the way back to `audio`. Deliberately does NOT use\n    ecapa_sim_eval.py\'s embed() (that version calls torchaudio.load from a file path and\n    wraps encode_batch in torch.no_grad() -- neither works here). Resample uses\n    torchaudio.functional.resample, which is a differentiable operation (unlike loading a\n    file), so gradient flows through the resample step too when native_sr != 16000.\n\n    SAMPLE-RATE ASSUMPTION FLAGGED FOR THE DIAGNOSTIC (pre-registration Sec 3, item 2):\n    native_sr defaults to args.surrogate_sample_rate (16000 in this script, matching the\n    dataset loaders\' sample_rate=16000) -- i.e. the assumption is that clean_audio/recon_wm/\n    perturbed are ALREADY at 16kHz and this is a no-op resample. What has NOT been verified\n    yet is cloned_output\'s native rate (YourTTS\'s own decoder output) -- --test_ecapa\'s\n    diagnostic print of audio.shape before/after resample is what confirms or falsifies this\n    before any calibration run trusts it.\n    """\n    if audio.dim() == 3:\n        audio = audio.squeeze(1)\n    if native_sr != 16000:\n        audio = torchaudio.functional.resample(audio, native_sr, 16000)\n    return model.encode_batch(audio).squeeze(1)\n\n\ndef pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,\n                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,\n                 lambda_kl_direct: float = 0.0, lambda_l1n_direct: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n'),
    ('                 lambda_kl_direct: float = 0.0, lambda_l1n_direct: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n', '                 lambda_kl_direct: float = 0.0, lambda_l1n_direct: float = 0.0,\n                 ecapa_model=None, lambda_ecapa_clone: float = 0.0,\n                 lambda_ecapa_direct: float = 0.0, ecapa_native_sr: int = 16000,\n                 noise_seed: int = None, verbose: bool = False):\n'),
    ('    Returns (recon_wm, perturbed_final, delta) -- all detached.\n    """\n', '    H-ENSEMBLE (2026-09-12 pre-registration, follow-up to H-SPEC/H-DIRECT):\n    lambda_ecapa_clone / lambda_ecapa_direct add ECAPA-TDNN (a second, architecturally\n    unrelated speaker encoder to YourTTS\'s own) into the DISRUPTION objective -- unlike\n    every prior lambda_* here, which only ever touched the watermark-preservation side.\n    ecapa_clone scores the same cloned_output the existing sim_loss already scores, via\n    a second encoder. ecapa_direct scores `perturbed` itself (pre-clone), motivated by\n    the conditioning-bandwidth finding that F5-TTS retains the reference audio almost\n    verbatim rather than regenerating it. Both reuse compute_sim_disruption_loss\n    unmodified, just fed ECAPA embeddings instead of YourTTS\'s. ecapa_model=None (the\n    default) makes both terms inert regardless of their lambdas, so this is a no-op\n    unless main() explicitly loads and passes an ECAPA model (see --test_ecapa).\n\n    Returns (recon_wm, perturbed_final, delta) -- all detached.\n    """\n'),
    ('    with torch.no_grad():\n        out = backbone.forward_full(clean_audio, message)\n        recon_wm = out["recon_wm"].detach()\n        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()\n', '    with torch.no_grad():\n        out = backbone.forward_full(clean_audio, message)\n        recon_wm = out["recon_wm"].detach()\n        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()\n        emb_clean_ecapa = (ecapa_embed(ecapa_model, clean_audio, ecapa_native_sr).detach()\n                           if ecapa_model is not None else None)\n'),
    ('        # backbone.model is in eval() mode (build_backbone calls .eval() -- correct\n', '        ecapa_loss_clone = None\n        ecapa_loss_direct = None\n        if ecapa_model is not None and (lambda_ecapa_clone > 0 or verbose):\n            emb_cloned_ecapa = ecapa_embed(ecapa_model, cloned_output, ecapa_native_sr)\n            ecapa_loss_clone = compute_sim_disruption_loss(emb_clean_ecapa, emb_cloned_ecapa)\n        if ecapa_model is not None and (lambda_ecapa_direct > 0 or verbose):\n            emb_perturbed_ecapa = ecapa_embed(ecapa_model, perturbed, ecapa_native_sr)\n            ecapa_loss_direct = compute_sim_disruption_loss(emb_clean_ecapa, emb_perturbed_ecapa)\n\n        # backbone.model is in eval() mode (build_backbone calls .eval() -- correct\n'),
    ('        grad_kl_direct = _add_term(kl_loss_direct, lambda_kl_direct)\n        grad_l1n_direct = _add_term(l1n_loss_direct, lambda_l1n_direct, retain=False)\n        grad_norm = grad.norm().item()\n', '        grad_kl_direct = _add_term(kl_loss_direct, lambda_kl_direct)\n        grad_l1n_direct = _add_term(l1n_loss_direct, lambda_l1n_direct)\n        grad_ecapa_clone = _add_term(ecapa_loss_clone, lambda_ecapa_clone)\n        grad_ecapa_direct = _add_term(ecapa_loss_direct, lambda_ecapa_direct, retain=False)\n        grad_norm = grad.norm().item()\n'),
    ('            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),\n                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n),\n                                       ("kl_direct", kl_loss_direct, grad_kl_direct),\n                                       ("l1n_direct", l1n_loss_direct, grad_l1n_direct)]:\n', '            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),\n                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n),\n                                       ("kl_direct", kl_loss_direct, grad_kl_direct),\n                                       ("l1n_direct", l1n_loss_direct, grad_l1n_direct),\n                                       ("ecapa_clone", ecapa_loss_clone, grad_ecapa_clone),\n                                       ("ecapa_direct", ecapa_loss_direct, grad_ecapa_direct)]:\n'),
    ('import torch\nimport torch.nn.functional as F\nimport soundfile as sf\n', 'import torch\nimport torch.nn.functional as F\nimport torchaudio\nimport soundfile as sf\n'),
    ('    p.add_argument("--lambda_l1n", type=float, default=0.0,\n                    help="H-SPEC (2026-09-12). Weight on compute_l1_to_noise (SafeSpeech\'s SPEC "\n                         "L1-to-noise term), same rationale as --lambda_kl. Default 0.0 = no change.")\n', '    p.add_argument("--lambda_l1n", type=float, default=0.0,\n                    help="H-SPEC (2026-09-12). Weight on compute_l1_to_noise (SafeSpeech\'s SPEC "\n                         "L1-to-noise term), same rationale as --lambda_kl. Default 0.0 = no change.")\n    p.add_argument("--lambda_ecapa_clone", type=float, default=0.0,\n                    help="H-ENSEMBLE (2026-09-12). Weight on compute_sim_disruption_loss evaluated "\n                         "with ECAPA-TDNN embeddings of clean_audio vs cloned_output -- the SAME "\n                         "clone the existing sim_loss already scores, judged by a second, "\n                         "architecturally unrelated encoder. Requires --test_ecapa (or nonzero here) "\n                         "to load the ECAPA model at all. Default 0.0 = no change to existing behavior.")\n    p.add_argument("--lambda_ecapa_direct", type=float, default=0.0,\n                    help="H-ENSEMBLE (2026-09-12). Weight on compute_sim_disruption_loss evaluated "\n                         "with ECAPA-TDNN embeddings of clean_audio vs `perturbed` itself (pre-clone) "\n                         "-- motivated by F5-TTS retaining the reference almost verbatim rather than "\n                         "regenerating it. Default 0.0 = no change to existing behavior.")\n    p.add_argument("--test_ecapa", action="store_true",\n                    help="H-ENSEMBLE diagnostic switch. Loads ECAPA-TDNN (speechbrain) and passes it "\n                         "into pgd_perturb so --diagnostic prints its gradient-norm ratio alongside "\n                         "every other term, even if --lambda_ecapa_clone/--lambda_ecapa_direct are "\n                         "left at 0.0. Run `pip install speechbrain` BEFORE this, then confirm "\n                         "build_backbone (which runs first in main(), below) still succeeds in THIS "\n                         "process before trusting anything this flag prints -- see "\n                         "load_ecapa_differentiable()\'s docstring for why that check matters.")\n'),
    ('    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=args.surrogate_sample_rate).to(device)\n', '    mel_fn = SafeSpeechMelSpectrogram(sampling_rate=args.surrogate_sample_rate).to(device)\n\n    ecapa_model = None\n    if args.test_ecapa or args.lambda_ecapa_clone > 0 or args.lambda_ecapa_direct > 0:\n        print("[main] Loading ECAPA-TDNN (H-ENSEMBLE)... build_backbone above already succeeded, "\n              "which is the canary check per pre-registration-hensemble-2026-09-12.md Sec 3.3 -- "\n              "if pip install speechbrain had broken the environment the way denoiser once did, "\n              "build_backbone would already have failed before reaching this line.")\n        ecapa_model = load_ecapa_differentiable(device)\n        print(f"[main] ECAPA-TDNN loaded on {device}.")\n'),
    ('        recon_wm, perturbed, delta = pgd_perturb(\n            backbone, surrogate, clean_audio, message, args.surrogate_text,\n            args.epsilon, step_size, n_steps=1, random_start=args.random_start,\n            lambda_wm=args.lambda_wm, mel_fn=mel_fn, lambda_kl=args.lambda_kl,\n            lambda_l1n=args.lambda_l1n, lambda_wm_clone=args.lambda_wm_clone,\n            noise_seed=123, verbose=True,\n        )\n', '        recon_wm, perturbed, delta = pgd_perturb(\n            backbone, surrogate, clean_audio, message, args.surrogate_text,\n            args.epsilon, step_size, n_steps=1, random_start=args.random_start,\n            lambda_wm=args.lambda_wm, mel_fn=mel_fn, lambda_kl=args.lambda_kl,\n            lambda_l1n=args.lambda_l1n, lambda_wm_clone=args.lambda_wm_clone,\n            ecapa_model=ecapa_model, lambda_ecapa_clone=args.lambda_ecapa_clone,\n            lambda_ecapa_direct=args.lambda_ecapa_direct,\n            ecapa_native_sr=args.surrogate_sample_rate,\n            noise_seed=123, verbose=True,\n        )\n'),
]


def _apply(path, edits):
    content = open(path).read()
    for old, new in edits:
        n = content.count(old)
        if n != 1:
            raise SystemExit(
                f"H-ENSEMBLE PATCH ABORTED (nothing written): anchor occurs {n} times "
                f"(need exactly 1). This usually means {path} does not yet have "
                f"H-SPEC+H-DIRECT applied, or has already had H-ENSEMBLE applied. "
                f"First 200 chars of the failing anchor:\n{old[:200]}"
            )
        content = content.replace(old, new, 1)
    open(path, "w").write(content)
    return content


def main():
    new_content = _apply(TARGET, EDITS)
    py_compile.compile(TARGET, doraise=True)
    ast.parse(new_content)
    for needle in ["def load_ecapa_differentiable(", "def ecapa_embed(", "ecapa_model=None",
                   "lambda_ecapa_clone", "lambda_ecapa_direct", "--test_ecapa"]:
        assert needle in new_content, f"MISSING after patch: {needle}"
    print(f"H-ENSEMBLE diagnostic patch applied to {TARGET} -- OK, "
          f"{len(new_content.splitlines())} lines, py_compile+ast.parse passed.")


if __name__ == "__main__":
    main()
