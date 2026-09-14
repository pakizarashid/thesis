"""
apply_clonereval_msgprocrank_patch.py -- wires --msgproc_lora_r into
cloner_watermark_eval.py (the F5-TTS / CosyVoice / MaskGCT cross-cloner script).

Unlike xtts_transfer_eval.py and save_audio_samples.py, this script was NOT
touched by apply_msgproc_rank_patch.py -- it has its own fully local
build_backbone(checkpoint_path, r=8, alpha=16) with no target_ranks support at
all, and is called from two places (--check_env's smoke test, and the main
eval path). Running it as-is against a checkpoint trained with
--msgproc_lora_r would hit the same shape-mismatch RuntimeError already seen
in train_route2_clone_aware.py, since msg_processor's LoRA would be built at
the wrong (default) rank before loading the checkpoint's reduced-rank tensors.

No change needed to the state-dict loading itself (same reasoning as the
disruption_pgd.py CLI patch): this script only ever loads a route2-trained
checkpoint's own lora_state_dict, saved at whatever rank the eval-side model
is built at here -- as long as --msgproc_lora_r matches the training-time
value, ranks match by construction and strict=False's normal missing/
unexpected-key tolerance is all that's needed.

Idempotent: safe to re-run (checked via a marker string).

Usage: python3 apply_clonereval_msgprocrank_patch.py <repo-root-or-.>
"""
import os
import sys

MARKER = "msgproc_lora_r wiring (2026-09-14): matches disruption_pgd.py /"


def _apply_unique(path, edits):
    """Each (old, new) must occur EXACTLY once."""
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if MARKER in content:
        print(f"[apply_clonereval_msgprocrank_patch] {path} already patched -- skipping.")
        return content, True
    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            raise SystemExit(
                f"ABORT: unique-edit {i} anchor occurs {count} times in {path} (expected 1). Nothing written.\n"
                f"----- anchor -----\n{old!r}"
            )
        new_content = new_content.replace(old, new, 1)
    return new_content, False


PATH = "src/eval/cloner_watermark_eval.py"

FN_OLD = (
    "def build_backbone(checkpoint_path, r=8, alpha=16):\n"
    "    backbone = VoiceMarkBackbone()\n"
    "    apply_lora_adapters(backbone, r=r, alpha=alpha)\n"
)
FN_NEW = (
    "def build_backbone(checkpoint_path, r=8, alpha=16, msgproc_lora_r=None):\n"
    "    # msgproc_lora_r wiring (2026-09-14): matches disruption_pgd.py /\n"
    "    # xtts_transfer_eval.py / save_audio_samples.py -- only needed to load a\n"
    "    # checkpoint trained with train_route2_clone_aware.py --msgproc_lora_r.\n"
    "    # MUST match the value used at training time or load_state_dict will hit\n"
    "    # a shape mismatch on msg_processor's LoRA tensors.\n"
    "    backbone = VoiceMarkBackbone()\n"
    "    target_ranks = {\"msg_processor\": msgproc_lora_r} if msgproc_lora_r else None\n"
    "    apply_lora_adapters(backbone, r=r, alpha=alpha, target_ranks=target_ranks)\n"
)

ARGS_OLD = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    "    args = p.parse_args()\n"
)
ARGS_NEW = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    p.add_argument("--msgproc_lora_r", type=int, default=None,\n'
    '                    help="Only needed to load a checkpoint trained with "\n'
    '                         "train_route2_clone_aware.py --msgproc_lora_r -- MUST match "\n'
    '                         "the value used at training time or load_state_dict will hit "\n'
    "                         \"a shape mismatch on msg_processor's LoRA tensors.\")\n"
    "    args = p.parse_args()\n"
)

CALL_OLD = "backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha)"
CALL_NEW = "backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha, msgproc_lora_r=args.msgproc_lora_r)"


def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."
    path = os.path.join(repo_root, PATH)

    content, already_done = _apply_unique(path, [(FN_OLD, FN_NEW), (ARGS_OLD, ARGS_NEW)])
    if already_done:
        return

    # CALL_OLD occurs twice (--check_env's smoke test + the main eval path) and
    # BOTH need the identical transformation -- replace both, but only after
    # confirming there are exactly 2, so an unexpected third occurrence (or a
    # structural change) aborts loudly instead of silently doing the wrong thing.
    count = content.count(CALL_OLD)
    if count != 2:
        raise SystemExit(
            f"ABORT: expected exactly 2 occurrences of the build_backbone call site in {path}, "
            f"found {count}. Nothing written.\n----- anchor -----\n{CALL_OLD!r}"
        )
    content = content.replace(CALL_OLD, CALL_NEW)

    with open(path, "w") as f:
        f.write(content)
    print(f"[apply_clonereval_msgprocrank_patch] Patched {path} (3 edits applied: function def, "
          f"CLI arg, {count} call sites).")


if __name__ == "__main__":
    main()
