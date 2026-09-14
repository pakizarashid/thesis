"""
apply_pgd_msgprocrank_cli_patch.py -- wires --msgproc_lora_r into disruption_pgd.py's
OWN command-line entry point (its main()), not just the build_backbone() function it
defines. apply_msgproc_rank_patch.py already added msgproc_lora_r as a parameter to
build_backbone() and propagated it correctly into xtts_transfer_eval.py and
save_audio_samples.py (both of which IMPORT build_backbone from this file) -- but it
never touched disruption_pgd.py's own argparse block or its own call to
build_backbone() inside main(), since at the time that patch was written, this
composability/PGD script wasn't yet the target of a msgproc_lora_r test.

Without this, running disruption_pgd.py directly (the dual-defense composability
check) against a checkpoint trained with --msgproc_lora_r would either:
  (a) crash on "unrecognized arguments: --msgproc_lora_r" (argparse doesn't know it), or
  (b) if the flag is simply omitted, silently build msg_processor's LoRA at the WRONG
      rank (defaulting to --lora_r, e.g. 8) and then crash on the same shape-mismatch
      RuntimeError already seen in train_route2_clone_aware.py, since the checkpoint's
      msg_processor tensors were saved at the reduced rank.

This patch adds the CLI flag and passes it through to build_backbone(), matching the
exact wiring pattern already used in xtts_transfer_eval.py / save_audio_samples.py.
No change needed to build_backbone()'s own state-dict loading (unlike
train_route2_clone_aware.py's build_trainable_backbone): this script and the other
eval scripts only ever load a route2-trained checkpoint's own lora_state_dict, which
was saved at whatever rank the eval-side model is built at here -- as long as
--msgproc_lora_r matches the training-time value (the caller's responsibility, same
as --lora_r always has been), ranks always match by construction and strict=False's
normal missing/unexpected-key tolerance is all that's needed.

Idempotent: safe to re-run (checked via a marker string).

Usage: python3 apply_pgd_msgprocrank_cli_patch.py <repo-root-or-.>
"""
import os
import sys

MARKER = "msgproc_lora_r wiring (2026-09-14): matches xtts_transfer_eval.py"


def _apply(path, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if MARKER in content:
        print(f"[apply_pgd_msgprocrank_cli_patch] {path} already patched -- skipping.")
        return
    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            raise SystemExit(
                f"ABORT: edit {i} anchor occurs {count} times in {path} (expected 1). Nothing written.\n"
                f"----- anchor -----\n{old!r}"
            )
        new_content = new_content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[apply_pgd_msgprocrank_cli_patch] Patched {path} ({len(edits)} edits applied).")


PGD_PATH = "src/eval/disruption_pgd.py"

OLD_ARGS = (
    '    p.add_argument("--capacity_lora_r", type=int, default=32)\n'
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
)
NEW_ARGS = (
    '    p.add_argument("--capacity_lora_r", type=int, default=32)\n'
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    p.add_argument("--msgproc_lora_r", type=int, default=None,\n'
    '                    help="msgproc_lora_r wiring (2026-09-14): matches xtts_transfer_eval.py -- "\n'
    '                         "only needed to load a checkpoint trained with "\n'
    '                         "train_route2_clone_aware.py --msgproc_lora_r. MUST match the value "\n'
    '                         "used at training time or load_state_dict will hit a shape mismatch "\n'
    '                         "on msg_processor\'s LoRA tensors.")\n'
)

OLD_CALL = (
    '    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,\n'
    '                               args.include_ffn, args.capacity_lora_r)\n'
)
NEW_CALL = (
    '    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,\n'
    '                               args.include_ffn, args.capacity_lora_r,\n'
    '                               msgproc_lora_r=args.msgproc_lora_r)\n'
)

_apply(PGD_PATH, [(OLD_ARGS, NEW_ARGS), (OLD_CALL, NEW_CALL)])
