"""
apply_hensemble_gen_patch.py (v2, 2026-10-14)

Adds H-ENSEMBLE switches to src/eval/composability_gen_protected_audio.py WITHOUT depending on the exact text of its generation loop:
  --lambda_ecapa_clone, --lambda_ecapa_direct   (default 0 -> behaviour unchanged)
  --noise_seed_base N                           (call k of pgd_perturb gets noise_seed N + k = N + clip index; default: not set)
It shadows the imported pgd_perturb with a thin wrapper that injects the ECAPA model/lambdas and the per-clip noise seed, so every call site
(diagnostic and full run) is covered. Also writes <save_dir>/_ensemble.json (lambdas) at the start of a run.
Needs apply_hensemble_patch_v2.py applied to disruption_pgd.py first (only matters when a lambda > 0 or noise_seed_base is set). Idempotent.
Usage: python patches/apply_hensemble_gen_patch.py [path]
"""
import ast, os, re, sys, py_compile

TARGET = sys.argv[1] if len(sys.argv) > 1 else "src/eval/composability_gen_protected_audio.py"
src = open(TARGET).read()
if "lambda_ecapa_clone" in src:
    print(f"[hensemble-gen patch] already applied to {TARGET}"); raise SystemExit(0)


def once(old, new, label):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"[hensemble-gen patch] ABORTED, nothing written: anchor '{label}' found {n}x (need 1):\n{old[:200]}")
    src = src.replace(old, new, 1)


# 1. CLI flags
once('    p.add_argument("--lambda_wm", type=float, default=1.0)\n',
     '    p.add_argument("--lambda_wm", type=float, default=1.0)\n'
     '    p.add_argument("--lambda_ecapa_clone", type=float, default=0.0,\n'
     '                   help="H-ENSEMBLE: weight of the ECAPA-TDNN disruption term on the YourTTS clone.")\n'
     '    p.add_argument("--lambda_ecapa_direct", type=float, default=0.0,\n'
     '                   help="H-ENSEMBLE: weight of the ECAPA-TDNN disruption term on the protected audio itself.")\n'
     '    p.add_argument("--noise_seed_base", type=int, default=None,\n'
     '                   help="If set, the k-th pgd_perturb call (= clip k) uses PGD random-start seed noise_seed_base + k, so arms share their random start.")\n',
     "--lambda_wm argument")

# 2. wrapper: inserted before the first top-level def (i.e. after the imports)
m = re.search(r"^def ", src, flags=re.M)
if not m:
    raise SystemExit("[hensemble-gen patch] ABORTED: no top-level def found")
WRAP = '''# ---- H-ENSEMBLE (2026-10-14): shadow pgd_perturb so every call gets the ECAPA terms and a per-clip random-start seed ----
_ENS_STATE = {"kw": {}, "base": None, "calls": 0}
_pgd_perturb_orig = pgd_perturb


def pgd_perturb(*a, **k):
    k.update(_ENS_STATE["kw"])
    if _ENS_STATE["base"] is not None:
        k["noise_seed"] = _ENS_STATE["base"] + _ENS_STATE["calls"]
    _ENS_STATE["calls"] += 1
    return _pgd_perturb_orig(*a, **k)


'''
src = src[:m.start()] + WRAP + src[m.start():]

# 3. configure the wrapper once the surrogate is loaded
once('    surrogate = load_yourtts_surrogate(device=device)\n',
     '    surrogate = load_yourtts_surrogate(device=device)\n'
     '    ens_kw = {}\n'
     '    if args.lambda_ecapa_clone > 0 or args.lambda_ecapa_direct > 0:\n'
     '        from disruption_pgd import load_ecapa_differentiable      # exists only after apply_hensemble_patch_v2.py\n'
     '        ens_kw = dict(ecapa_model=load_ecapa_differentiable(device), lambda_ecapa_clone=args.lambda_ecapa_clone,\n'
     '                      lambda_ecapa_direct=args.lambda_ecapa_direct, ecapa_native_sr=16000)\n'
     '        print(f"[main] H-ENSEMBLE on: lambda_ecapa_clone={args.lambda_ecapa_clone} lambda_ecapa_direct={args.lambda_ecapa_direct}")\n'
     '    _ENS_STATE.update(kw=ens_kw, base=args.noise_seed_base, calls=0)\n'
     '    if args.save_dir:\n'
     '        os.makedirs(args.save_dir, exist_ok=True)\n'
     '        with open(os.path.join(args.save_dir, "_ensemble.json"), "w") as _f:\n'
     '            json.dump({"lambda_ecapa_clone": args.lambda_ecapa_clone, "lambda_ecapa_direct": args.lambda_ecapa_direct,\n'
     '                       "noise_seed_base": args.noise_seed_base, "epsilon": args.epsilon, "n_steps": args.n_steps,\n'
     '                       "lambda_wm": args.lambda_wm}, _f, indent=2)\n',
     "surrogate load")

ast.parse(src)
open(TARGET, "w").write(src)
py_compile.compile(TARGET, doraise=True)
print(f"[hensemble-gen patch] applied to {TARGET}")
