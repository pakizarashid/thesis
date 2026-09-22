"""
apply_hdirect_patch.py -- H-DIRECT (2026-09-12) code update, STANDALONE / catch-up variant.

Use this ONLY if your files already have the H-SPEC patch applied (lambda_kl,
lambda_l1n, lambda_wm_clone already present) but NOT yet the H-DIRECT patch
(lambda_kl_direct, lambda_l1n_direct). This is the situation your currently-
running Kaggle session is in: you patched with the H-SPEC-only version
earlier, and re-running the combined patch script from scratch would fail
(its H-SPEC-edit anchors expect the ORIGINAL unpatched file, which yours no
longer is). This script's anchors instead expect the H-SPEC-patched file, so
it applies cleanly on top of your current state without needing to re-clone
or re-patch anything.

Usage (run once, from the repo root, in your current session):
    python apply_hdirect_patch.py src/eval/disruption_pgd.py src/eval/demucs_fallback_eval.py

Each edit is checked to occur EXACTLY ONCE in its target file before being
applied. If a file doesn't match (e.g. it's still fully original, or already
has H-DIRECT applied), it fails loudly naming which edit didn't match, and
leaves that file untouched.
"""
import sys

EDITS_HDIRECT_PGD = [
    # signature: add lambda_kl_direct/lambda_l1n_direct params
    ('def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,\n                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n',
     'def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n                 lambda_wm: float = 0.0, mel_fn=None, lambda_kl: float = 0.0,\n                 lambda_l1n: float = 0.0, lambda_wm_clone: float = 0.0,\n                 lambda_kl_direct: float = 0.0, lambda_l1n_direct: float = 0.0,\n                 noise_seed: int = None, verbose: bool = False):\n'),
    # docstring: explain H-DIRECT rationale
    ("    to this project's YourTTS surrogate. mel_fn is REQUIRED if either is\n    nonzero. random_noise is fixed once per utterance (not resampled every\n    step); noise_seed makes it reproducible.\n\n    Returns (recon_wm, perturbed_final, delta) -- all detached.\n",
     "    to this project's YourTTS surrogate. mel_fn is REQUIRED if either is\n    nonzero. random_noise is fixed once per utterance (not resampled every\n    step); noise_seed makes it reproducible.\n\n    H-DIRECT (2026-09-12, follow-up to H-SPEC): lambda_kl_direct / lambda_l1n_direct\n    apply the SAME SafeSpeech SPEC terms directly to the PERTURBED INPUT's own mel\n    spectrogram (`perturbed`, i.e. recon_wm + delta -- BEFORE the surrogate clones it),\n    instead of to the surrogate's generated output. Motivation: the project's own\n    five-architecture ladder (research-question-answered doc) established that F5-TTS\n    does not regenerate the reference through a decoder -- it RETAINS the reference mel\n    almost verbatim. H-SPEC's kl_loss/l1n_loss (above) push the YourTTS SURROGATE's\n    generated clone toward noise -- a signal that only matters if F5-TTS's own\n    conditioning behaves like the surrogate's, which the eps-sweep and H-SPEC results\n    both suggest it does not. Pushing the INPUT itself toward noise-like mel statistics\n    requires no surrogate forward pass for this term at all, so it cannot fail to\n    transfer to F5-TTS for the same reason H-SPEC's kl_loss did -- it attacks the thing\n    F5-TTS actually copies, not a YourTTS-mediated proxy for it. Still a hypothesis, not\n    a promise: this may simply behave like brute-force epsilon (same quality cost), which\n    is a legitimate, reportable outcome too. random_noise_direct is independent of\n    random_noise (different shape: perturbed's, not cloned_output's) but drawn from the\n    same noise_gen when noise_seed is set, so a given noise_seed remains fully\n    reproducible across all four SPEC-style terms.\n\n    Returns (recon_wm, perturbed_final, delta) -- all detached.\n"),
    # validate mel_fn requirement covers new lambdas
    ('    if (lambda_kl > 0 or lambda_l1n > 0) and mel_fn is None:\n        raise ValueError("pgd_perturb: lambda_kl/lambda_l1n > 0 requires mel_fn (pass the "\n                          "SafeSpeechMelSpectrogram instance already built in main()).")\n',
     '    if (lambda_kl > 0 or lambda_l1n > 0 or lambda_kl_direct > 0 or lambda_l1n_direct > 0) and mel_fn is None:\n        raise ValueError("pgd_perturb: lambda_kl/lambda_l1n/lambda_kl_direct/lambda_l1n_direct > 0 "\n                          "requires mel_fn (pass the SafeSpeechMelSpectrogram instance already "\n                          "built in main()).")\n'),
    # init random_noise_direct alongside random_noise
    ('    random_noise = None\n    noise_gen = None\n    if noise_seed is not None:\n        noise_gen = torch.Generator(device=delta.device).manual_seed(noise_seed)\n',
     '    random_noise = None\n    random_noise_direct = None\n    noise_gen = None\n    if noise_seed is not None:\n        noise_gen = torch.Generator(device=delta.device).manual_seed(noise_seed)\n'),
    # compute kl_loss_direct/l1n_loss_direct on perturbed input
    ('        kl_loss = None\n        l1n_loss = None\n        if lambda_kl > 0 or lambda_l1n > 0:\n            if random_noise is None or random_noise.shape != cloned_output.shape:\n                if noise_gen is not None:\n                    random_noise = torch.randn(cloned_output.shape, generator=noise_gen,\n                                                device=cloned_output.device)\n                else:\n                    random_noise = torch.randn_like(cloned_output)\n            if lambda_kl > 0:\n                kl_loss = compute_kl_to_noise(mel_fn, cloned_output, random_noise)\n            if lambda_l1n > 0:\n                l1n_loss = compute_l1_to_noise(mel_fn, cloned_output, random_noise)\n',
     '        kl_loss = None\n        l1n_loss = None\n        if lambda_kl > 0 or lambda_l1n > 0:\n            if random_noise is None or random_noise.shape != cloned_output.shape:\n                if noise_gen is not None:\n                    random_noise = torch.randn(cloned_output.shape, generator=noise_gen,\n                                                device=cloned_output.device)\n                else:\n                    random_noise = torch.randn_like(cloned_output)\n            if lambda_kl > 0:\n                kl_loss = compute_kl_to_noise(mel_fn, cloned_output, random_noise)\n            if lambda_l1n > 0:\n                l1n_loss = compute_l1_to_noise(mel_fn, cloned_output, random_noise)\n\n        kl_loss_direct = None\n        l1n_loss_direct = None\n        if lambda_kl_direct > 0 or lambda_l1n_direct > 0:\n            if random_noise_direct is None or random_noise_direct.shape != perturbed.shape:\n                if noise_gen is not None:\n                    random_noise_direct = torch.randn(perturbed.shape, generator=noise_gen,\n                                                       device=perturbed.device)\n                else:\n                    random_noise_direct = torch.randn_like(perturbed)\n            if lambda_kl_direct > 0:\n                kl_loss_direct = compute_kl_to_noise(mel_fn, perturbed, random_noise_direct)\n            if lambda_l1n_direct > 0:\n                l1n_loss_direct = compute_l1_to_noise(mel_fn, perturbed, random_noise_direct)\n'),
    # accumulate grad_kl_direct/grad_l1n_direct
    ('        grad_wm = _add_term(wm_loss, lambda_wm)\n        grad_wm_clone = _add_term(wm_loss_clone, lambda_wm_clone)\n        grad_kl = _add_term(kl_loss, lambda_kl)\n        grad_l1n = _add_term(l1n_loss, lambda_l1n, retain=False)\n        grad_norm = grad.norm().item()\n',
     '        grad_wm = _add_term(wm_loss, lambda_wm)\n        grad_wm_clone = _add_term(wm_loss_clone, lambda_wm_clone)\n        grad_kl = _add_term(kl_loss, lambda_kl)\n        grad_l1n = _add_term(l1n_loss, lambda_l1n)\n        grad_kl_direct = _add_term(kl_loss_direct, lambda_kl_direct)\n        grad_l1n_direct = _add_term(l1n_loss_direct, lambda_l1n_direct, retain=False)\n        grad_norm = grad.norm().item()\n'),
    # verbose diagnostic prints new terms
    ('        if verbose:\n            parts = [f"sim_loss={sim_loss.item():.4f} |grad_sim|={grad_sim_for_log.norm().item():.4e}"]\n            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),\n                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n)]:\n',
     '        if verbose:\n            parts = [f"sim_loss={sim_loss.item():.4f} |grad_sim|={grad_sim_for_log.norm().item():.4e}"]\n            for name, loss_val, g in [("wm", wm_loss, grad_wm), ("wm_clone", wm_loss_clone, grad_wm_clone),\n                                       ("kl", kl_loss, grad_kl), ("l1n", l1n_loss, grad_l1n),\n                                       ("kl_direct", kl_loss_direct, grad_kl_direct),\n                                       ("l1n_direct", l1n_loss_direct, grad_l1n_direct)]:\n'),
]

EDITS_HDIRECT_STAGE0 = [
    # CLI: add --lambda_kl_direct/--lambda_l1n_direct
    ('    p.add_argument("--lambda_l1n", type=float, default=0.0,\n                   help="H-SPEC (2026-09-12). SafeSpeech\'s SPEC L1-to-noise term, same "\n                        "rationale as --lambda_kl. Default 0.0 = no change.")\n',
     '    p.add_argument("--lambda_l1n", type=float, default=0.0,\n                   help="H-SPEC (2026-09-12). SafeSpeech\'s SPEC L1-to-noise term, same "\n                        "rationale as --lambda_kl. Default 0.0 = no change.")\n    p.add_argument("--lambda_kl_direct", type=float, default=0.0,\n                   help="H-DIRECT (2026-09-12, follow-up to H-SPEC). SafeSpeech\'s SPEC "\n                        "KL-to-noise term applied to the PERTURBED INPUT\'s own mel "\n                        "(before cloning), not the surrogate\'s cloned output -- bypasses "\n                        "the YourTTS surrogate for this term entirely, so it cannot fail to "\n                        "transfer to F5-TTS the way lambda_kl did. Default 0.0 = no change.")\n    p.add_argument("--lambda_l1n_direct", type=float, default=0.0,\n                   help="H-DIRECT (2026-09-12). SafeSpeech\'s SPEC L1-to-noise term, same "\n                        "input-mel-direct rationale as --lambda_kl_direct. Default 0.0 = no change.")\n'),
    # diagnostic pgd_perturb call: thread new lambdas
    ('            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone, noise_seed=123,\n        )\n',
     '            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone,\n            lambda_kl_direct=args.lambda_kl_direct, lambda_l1n_direct=args.lambda_l1n_direct,\n            noise_seed=123,\n        )\n'),
    # print header: show new lambdas
    ('    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "\n          f"lambda_wm={args.lambda_wm} lambda_kl={args.lambda_kl} lambda_l1n={args.lambda_l1n} "\n          f"lambda_wm_clone={args.lambda_wm_clone} | n={args.n_utterances}")\n',
     '    print(f"DEMUCS FALLBACK EVAL | backend={backend} | eps={args.epsilon} | "\n          f"lambda_wm={args.lambda_wm} lambda_kl={args.lambda_kl} lambda_l1n={args.lambda_l1n} "\n          f"lambda_wm_clone={args.lambda_wm_clone} lambda_kl_direct={args.lambda_kl_direct} "\n          f"lambda_l1n_direct={args.lambda_l1n_direct} | n={args.n_utterances}")\n'),
    # full-run pgd_perturb call: thread new lambdas
    ('            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone, noise_seed=123 + i,\n        )\n',
     '            mel_fn=mel_fn, lambda_kl=args.lambda_kl, lambda_l1n=args.lambda_l1n,\n            lambda_wm_clone=args.lambda_wm_clone,\n            lambda_kl_direct=args.lambda_kl_direct, lambda_l1n_direct=args.lambda_l1n_direct,\n            noise_seed=123 + i,\n        )\n'),
    # output JSON: record new lambdas
    ('                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,\n                        "lambda_wm": args.lambda_wm, "step_size": step_size,\n                        "lambda_kl": args.lambda_kl, "lambda_l1n": args.lambda_l1n,\n                        "lambda_wm_clone": args.lambda_wm_clone},\n',
     '                "pgd": {"epsilon": args.epsilon, "n_steps": args.n_steps,\n                        "lambda_wm": args.lambda_wm, "step_size": step_size,\n                        "lambda_kl": args.lambda_kl, "lambda_l1n": args.lambda_l1n,\n                        "lambda_wm_clone": args.lambda_wm_clone,\n                        "lambda_kl_direct": args.lambda_kl_direct,\n                        "lambda_l1n_direct": args.lambda_l1n_direct},\n'),
]


def _apply(path, edits):
    with open(path, "r") as f:
        content = f.read()
    for i, (old, new) in enumerate(edits):
        count = content.count(old)
        if count != 1:
            print(f"FAILED on {path}, edit {i+1}/{len(edits)}: expected 1 occurrence of the "
                  f"anchor text, found {count}. This file has diverged from what this patch "
                  f"expects -- stopping WITHOUT writing {path} (it is untouched). Paste this "
                  f"message back to Claude with the current content of {path} to regenerate "
                  f"the patch.")
            return False
        content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"OK: applied all {len(edits)} edits to {path}.")
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python apply_hdirect_patch.py <path to disruption_pgd.py> "
              "<path to demucs_fallback_eval.py>")
        sys.exit(1)
    pgd_path, stage0_path = sys.argv[1], sys.argv[2]
    ok_pgd = _apply(pgd_path, EDITS_HDIRECT_PGD)
    ok_stage0 = _apply(stage0_path, EDITS_HDIRECT_STAGE0) if ok_pgd else False
    if not (ok_pgd and ok_stage0):
        sys.exit(1)
    print("H-DIRECT catch-up patch fully applied to both files.")


if __name__ == "__main__":
    main()
