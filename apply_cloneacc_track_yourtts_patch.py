"""
apply_cloneacc_track_yourtts_patch.py -- gen_samples_yourtts.py only saved audio
(built for CARRIER-PROBE's generation step); adds clone-ACC tracking so a
--layer_scales run directly reports whether the reweighted carrier changed
attribution ACC, the same way cloner_watermark_eval.py already does for the
other architectures.
"""
import os, sys

_DEFAULT_TARGET = "src/eval/gen_samples_yourtts.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

OLD = (
    '    os.makedirs(args.save_clones_dir, exist_ok=True)\n'
    '    n_written = 0\n'
    '    for i, batch in enumerate(loader):\n'
)
NEW = (
    '    os.makedirs(args.save_clones_dir, exist_ok=True)\n'
    '    n_written = 0\n'
    '    accs_clone = []\n'
    '    for i, batch in enumerate(loader):\n'
)

OLD2 = (
    '            cloned = surrogate.clone_voice(recon_wm, text=text)\n'
)
NEW2 = (
    '            cloned = surrogate.clone_voice(recon_wm, text=text)\n'
    '            a_cln = detect_acc(backbone, cloned, message)\n'
    '            accs_clone.append(a_cln)\n'
    '            print(f"[gen_samples_yourtts] [{i}] acc_yourtts_clone={a_cln:.4f}")\n'
)

OLD3 = (
    '    print(f"[gen_samples_yourtts] wrote {n_written} reference/clone pairs to {args.save_clones_dir}")\n'
)
NEW3 = (
    '    print(f"[gen_samples_yourtts] wrote {n_written} reference/clone pairs to {args.save_clones_dir}")\n'
    '    if accs_clone:\n'
    '        mean_acc = sum(accs_clone) / len(accs_clone)\n'
    '        print(f"[gen_samples_yourtts] RESULT mean acc_yourtts_clone={mean_acc:.4f} (n={len(accs_clone)}, "\n'
    '              f"layer_scales={layer_scales})")\n'
)

def _apply(path, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if "accs_clone" in content:
        print(f"[apply_cloneacc_track_yourtts_patch] {path} already patched -- skipping.")
        return
    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            raise SystemExit(f"ABORT: edit {i} anchor occurs {count} times in {path} (expected 1). Nothing written.\n{old[:300]!r}")
        new_content = new_content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[apply_cloneacc_track_yourtts_patch] Patched {path} ({len(edits)} edits applied).")

if __name__ == "__main__":
    _apply(TARGET, [(OLD, NEW), (OLD2, NEW2), (OLD3, NEW3)])
