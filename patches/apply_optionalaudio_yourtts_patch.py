"""
apply_optionalaudio_yourtts_patch.py -- makes --save_clones_dir optional in
gen_samples_yourtts.py. CARRIER-REWEIGHT only needs the printed
acc_yourtts_clone numbers (already computed in-memory via detect_acc), not the
audio itself -- writing 30 .wav files per run (2 x n_utterances) was pure disk
cost with no purpose for this experiment. Omit --save_clones_dir entirely to
skip all disk writes; pass it only when you actually want to keep/listen to the
clones for some other reason.
"""
import os, sys

_DEFAULT_TARGET = "src/eval/gen_samples_yourtts.py"
TARGET = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TARGET
if not os.path.exists(TARGET) and os.path.exists(os.path.basename(TARGET)):
    TARGET = os.path.basename(TARGET)

ARG_OLD = '    p.add_argument("--save_clones_dir", type=str, required=True)\n'
ARG_NEW = (
    '    p.add_argument("--save_clones_dir", type=str, default=None,\n'
    '                    help="Omit to skip writing audio to disk entirely -- "\n'
    '                         "CARRIER-REWEIGHT only needs the printed "\n'
    '                         "acc_yourtts_clone numbers, not the audio itself. Pass "\n'
    '                         "a path only when you actually want to keep the clones.")\n'
)

MAKEDIRS_OLD = '    os.makedirs(args.save_clones_dir, exist_ok=True)\n'
MAKEDIRS_NEW = (
    '    if args.save_clones_dir:\n'
    '        os.makedirs(args.save_clones_dir, exist_ok=True)\n'
)

WRITE_OLD = (
    '        sf.write(os.path.join(args.save_clones_dir, f"sample{i}_reference.wav"),\n'
    '                 recon_wm[0].detach().cpu().reshape(-1).numpy(), 16000)\n'
    '        sf.write(os.path.join(args.save_clones_dir, f"sample{i}_clone_yourtts.wav"),\n'
    '                 cloned[0].detach().cpu().reshape(-1).numpy(), 16000)\n'
    '        n_written += 1\n'
)
WRITE_NEW = (
    '        if args.save_clones_dir:\n'
    '            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_reference.wav"),\n'
    '                     recon_wm[0].detach().cpu().reshape(-1).numpy(), 16000)\n'
    '            sf.write(os.path.join(args.save_clones_dir, f"sample{i}_clone_yourtts.wav"),\n'
    '                     cloned[0].detach().cpu().reshape(-1).numpy(), 16000)\n'
    '        n_written += 1\n'
)

FINAL_OLD = '    print(f"[gen_samples_yourtts] wrote {n_written} reference/clone pairs to {args.save_clones_dir}")\n'
FINAL_NEW = (
    '    if args.save_clones_dir:\n'
    '        print(f"[gen_samples_yourtts] wrote {n_written} reference/clone pairs to {args.save_clones_dir}")\n'
    '    else:\n'
    '        print(f"[gen_samples_yourtts] processed {n_written} utterances (audio not saved -- "\n'
    '              f"pass --save_clones_dir to keep the .wav files).")\n'
)


def _apply(path, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if "audio not saved" in content:
        print(f"[apply_optionalaudio_yourtts_patch] {path} already patched -- skipping.")
        return
    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            raise SystemExit(f"ABORT: edit {i} anchor occurs {count} times in {path} (expected 1). Nothing written.\n{old[:300]!r}")
        new_content = new_content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[apply_optionalaudio_yourtts_patch] Patched {path} ({len(edits)} edits applied).")


if __name__ == "__main__":
    _apply(TARGET, [
        (ARG_OLD, ARG_NEW),
        (MAKEDIRS_OLD, MAKEDIRS_NEW),
        (WRITE_OLD, WRITE_NEW),
        (FINAL_OLD, FINAL_NEW),
    ])
