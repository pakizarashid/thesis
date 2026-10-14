"""
apply_hensemble_patch_v2.py (2026-10-14) -- H-ENSEMBLE for the CURRENT main-branch src/eval/disruption_pgd.py.

The September patch (apply_hensemble_patch.py) assumed H-SPEC + H-DIRECT were already merged (pgd_perturb with mel_fn, lambda_kl, ..., noise_seed).
They closed null and are not on main: main's pgd_perturb is  (backbone, surrogate, clean_audio, message, text, epsilon, step_size, n_steps,
random_start, lambda_wm=0.0, verbose=False).  This patch targets exactly that, and changes only pgd_perturb:

  * load_ecapa_differentiable(device), ecapa_embed(model, audio, native_sr)   new helpers (ECAPA-TDNN, gradient-carrying)
  * pgd_perturb(..., ecapa_model=None, lambda_ecapa_clone=0.0, lambda_ecapa_direct=0.0, ecapa_native_sr=16000, noise_seed=None, verbose=False)
      - ecapa_clone term : compute_sim_disruption_loss(ECAPA(clean), ECAPA(YourTTS clone of perturbed))
      - ecapa_direct term: compute_sim_disruption_loss(ECAPA(clean), ECAPA(perturbed))
      - noise_seed       : reproducible random start (None = the old torch.rand_like behaviour, unchanged)
      - with ecapa_model=None and noise_seed=None the function behaves exactly as before.
No CLI flags are added (the generator composability_gen_protected_audio.py drives it). Idempotent. Usage: python patches/apply_hensemble_patch_v2.py [path]
"""
import ast, os, sys, py_compile

TARGET = sys.argv[1] if len(sys.argv) > 1 else "src/eval/disruption_pgd.py"

HELPERS = '''def load_ecapa_differentiable(device):
    """
    H-ENSEMBLE (2026-10-14 addendum). ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) loaded for use INSIDE the PGD loop. Unlike
    ecapa_sim_eval.py (separate process, torch.no_grad on purpose) gradients must flow here, so encode_batch is never wrapped in no_grad.
    """
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def ecapa_embed(model, audio: torch.Tensor, native_sr: int):
    """Differentiable ECAPA embedding of an in-memory waveform batch [B, T] or [B, 1, T]; resample (if needed) is inside the autograd graph."""
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    if native_sr != 16000:
        audio = torchaudio.functional.resample(audio, native_sr, 16000)
    return model.encode_batch(audio).squeeze(1)


'''

EDITS = [
    # 0 import
    ('import torch\nimport torch.nn.functional as F\nimport soundfile as sf\n',
     'import torch\nimport torch.nn.functional as F\nimport torchaudio\nimport soundfile as sf\n'),
    # 1 helpers + signature
    ('def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n'
     '                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n'
     '                 lambda_wm: float = 0.0, verbose: bool = False):\n',
     HELPERS +
     'def pgd_perturb(backbone, surrogate, clean_audio: torch.Tensor, message: torch.Tensor, text: str,\n'
     '                 epsilon: float, step_size: float, n_steps: int, random_start: bool,\n'
     '                 lambda_wm: float = 0.0, verbose: bool = False,\n'
     '                 ecapa_model=None, lambda_ecapa_clone: float = 0.0, lambda_ecapa_direct: float = 0.0,\n'
     '                 ecapa_native_sr: int = 16000, noise_seed: int = None):\n'),
    # 2 clean ECAPA embedding + reproducible random start
    ('        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()\n\n'
     '    if random_start:\n'
     '        delta = (torch.rand_like(recon_wm) * 2 - 1) * epsilon\n',
     '        emb_clean = surrogate.compute_speaker_embedding(clean_audio).detach()\n'
     '        emb_clean_ecapa = (ecapa_embed(ecapa_model, clean_audio, ecapa_native_sr).detach()\n'
     '                           if ecapa_model is not None else None)\n\n'
     '    if random_start:\n'
     '        if noise_seed is not None:      # H-ENSEMBLE: same random start in every arm of a paired comparison\n'
     '            _g = torch.Generator(device=recon_wm.device).manual_seed(int(noise_seed))\n'
     '            delta = (torch.rand(recon_wm.shape, generator=_g, device=recon_wm.device, dtype=recon_wm.dtype) * 2 - 1) * epsilon\n'
     '        else:\n'
     '            delta = (torch.rand_like(recon_wm) * 2 - 1) * epsilon\n'),
    # 3 ECAPA loss terms
    ('        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)\n'
     '        sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)\n\n'
     '        # backbone.model is in eval() mode (build_backbone calls .eval() -- correct\n',
     '        emb_cloned = surrogate.compute_speaker_embedding(cloned_output)\n'
     '        sim_loss = compute_sim_disruption_loss(emb_clean, emb_cloned)\n\n'
     '        # H-ENSEMBLE: second speaker encoder in the disruption objective (inert unless ecapa_model is given)\n'
     '        ecapa_loss_clone = ecapa_loss_direct = None\n'
     '        if ecapa_model is not None and (lambda_ecapa_clone > 0 or verbose):\n'
     '            ecapa_loss_clone = compute_sim_disruption_loss(emb_clean_ecapa, ecapa_embed(ecapa_model, cloned_output, ecapa_native_sr))\n'
     '        if ecapa_model is not None and (lambda_ecapa_direct > 0 or verbose):\n'
     '            ecapa_loss_direct = compute_sim_disruption_loss(emb_clean_ecapa, ecapa_embed(ecapa_model, perturbed, ecapa_native_sr))\n\n'
     '        # backbone.model is in eval() mode (build_backbone calls .eval() -- correct\n'),
    # 4 gradients
    ('        grad_wm = torch.autograd.grad(wm_loss, delta, retain_graph=False, create_graph=False)[0]\n'
     '        grad = grad_sim + lambda_wm * grad_wm\n'
     '        grad_norm = grad.norm().item()\n',
     '        _want_c, _want_d = ecapa_loss_clone is not None, ecapa_loss_direct is not None\n'
     '        grad_wm = torch.autograd.grad(wm_loss, delta, retain_graph=(_want_c or _want_d), create_graph=False)[0]\n'
     '        grad_ec_c = torch.autograd.grad(ecapa_loss_clone, delta, retain_graph=_want_d, create_graph=False)[0] if _want_c else None\n'
     '        grad_ec_d = torch.autograd.grad(ecapa_loss_direct, delta, retain_graph=False, create_graph=False)[0] if _want_d else None\n'
     '        grad = grad_sim + lambda_wm * grad_wm\n'
     '        if lambda_ecapa_clone > 0 and grad_ec_c is not None:\n'
     '            grad = grad + lambda_ecapa_clone * grad_ec_c\n'
     '        if lambda_ecapa_direct > 0 and grad_ec_d is not None:\n'
     '            grad = grad + lambda_ecapa_direct * grad_ec_d\n'
     '        grad_norm = grad.norm().item()\n'),
    # 5 verbose print of the ECAPA terms
    ('        if step == 0 and grad_norm == 0.0:\n',
     '        if verbose and (grad_ec_c is not None or grad_ec_d is not None):\n'
     '            for _nm, _l, _g in (("ecapa_clone", ecapa_loss_clone, grad_ec_c), ("ecapa_direct", ecapa_loss_direct, grad_ec_d)):\n'
     '                if _g is not None:\n'
     '                    print(f"  [pgd step {step}] {_nm}_loss={_l.item():.4f} |grad_{_nm}|={_g.norm().item():.4e} "\n'
     '                          f"(sim:{_nm} raw ratio={grad_sim.norm().item() / max(_g.norm().item(), 1e-12):.2f})")\n'
     '        if step == 0 and grad_norm == 0.0:\n'),
]


def _apply(path, edits):
    content = open(path).read()
    if "lambda_ecapa_clone" in content:
        print(f"[hensemble v2] already applied to {path}"); return None
    for k, (old, new) in enumerate(edits):
        n = content.count(old)
        if n != 1:
            raise SystemExit(f"[hensemble v2] ABORTED, nothing written: anchor {k} occurs {n}x (need exactly 1):\n{old[:240]}")
        content = content.replace(old, new, 1)
    return content


def main():
    new = _apply(TARGET, EDITS)
    if new is None:
        return
    ast.parse(new)
    open(TARGET, "w").write(new)
    py_compile.compile(TARGET, doraise=True)
    for needle in ["def load_ecapa_differentiable(", "def ecapa_embed(", "ecapa_model=None", "lambda_ecapa_direct", "noise_seed: int = None"]:
        assert needle in new, f"MISSING after patch: {needle}"
    print(f"[hensemble v2] applied to {TARGET} ({len(new.splitlines())} lines, py_compile + ast.parse passed)")


if __name__ == "__main__":
    main()
