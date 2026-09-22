"""
apply_msgproc_rank_patch.py -- wires up an independent, lower LoRA rank for
msg_processor ONLY (detector stays at --lora_r), as a targeted test of the
quality/SIM trade-off found in --train_msg_processor runs today. Two
independent attempts to fix it via --epochs and --lambda_clone both landed
statistically identical to the untouched run (paired p=0.907 and p=0.606 on
ACC/SIM respectively) -- neither training duration nor loss weighting moved
the PESQ/SIM cost at all, which points at something structural to the LoRA
capacity itself, not at how hard/long the clone objective is optimized.

apply_lora_adapters() already has a capacity-override mechanism (ffn_r /
ffn_targets), but it's coupled to include_ffn=True -- using it also turns on
FFN-layer LoRA wrapping, not just a smaller attention-only rank. That's a
confound for this specific test (we want ONLY rank to change, not
architecture). This patch adds a clean target_ranks dict that overrides rank
per-target independent of FFN, then wires --msgproc_lora_r through
train_route2_clone_aware.py (training) and disruption_pgd.py's build_backbone
(the eval-side helper shared by xtts_transfer_eval.py / save_audio_samples.py
/ gen_samples_yourtts.py) so a checkpoint trained with a reduced msg_processor
rank can also be correctly reconstructed (matching shapes) at eval time --
otherwise load_state_dict would hit a shape mismatch, since eval scripts
currently always rebuild msg_processor's LoRA at the same rank as detector.

Idempotent: safe to re-run on an already-patched file (checked via a marker
string unique to each edit).
"""
import os, sys

def _apply(path, marker, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if marker in content:
        print(f"[apply_msgproc_rank_patch] {path} already patched -- skipping.")
        return
    new_content = content
    for i, (old, new) in enumerate(edits):
        count = new_content.count(old)
        if count != 1:
            raise SystemExit(f"ABORT: edit {i} anchor occurs {count} times in {path} (expected 1). Nothing written.\n{old[:300]!r}")
        new_content = new_content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(new_content)
    print(f"[apply_msgproc_rank_patch] Patched {path} ({len(edits)} edits applied).")


# ---------------------------------------------------------------- adapters.py
ADAPTERS_PATH = "src/models/adapters.py"

ADAPTERS_OLD_SIG = (
    'def apply_lora_adapters(backbone, r: int = 8, alpha: int = 16, targets=("msg_processor", "detector"),\n'
    '                         include_ffn: bool = False, ffn_r: int = None, ffn_targets: tuple = None):\n'
)
ADAPTERS_NEW_SIG = (
    'def apply_lora_adapters(backbone, r: int = 8, alpha: int = 16, targets=("msg_processor", "detector"),\n'
    '                         include_ffn: bool = False, ffn_r: int = None, ffn_targets: tuple = None,\n'
    '                         target_ranks: dict = None):\n'
)

ADAPTERS_OLD_RANK = (
    '        this_include_ffn = include_ffn and (target_name in ffn_targets)\n'
    '        this_r = ffn_r if this_include_ffn else r\n'
)
ADAPTERS_NEW_RANK = (
    '        this_include_ffn = include_ffn and (target_name in ffn_targets)\n'
    '        if target_ranks and target_name in target_ranks:\n'
    '            # 2026-09-14 quality-tradeoff patch: independent rank override per\n'
    '            # target, decoupled from include_ffn -- unlike ffn_r, this does NOT\n'
    '            # also turn on FFN-layer wrapping, so it isolates capacity alone.\n'
    '            this_r = target_ranks[target_name]\n'
    '        else:\n'
    '            this_r = ffn_r if this_include_ffn else r\n'
)

_apply(ADAPTERS_PATH, "target_ranks: dict = None", [
    (ADAPTERS_OLD_SIG, ADAPTERS_NEW_SIG),
    (ADAPTERS_OLD_RANK, ADAPTERS_NEW_RANK),
])


# ------------------------------------------------------- disruption_pgd.py
# (shared build_backbone() used by xtts_transfer_eval.py, save_audio_samples.py,
#  gen_samples_yourtts.py -- patching here propagates to all of them)
PGD_PATH = "src/eval/disruption_pgd.py"

PGD_OLD_SIG = 'def build_backbone(checkpoint_path: str, r: int, alpha: int, include_ffn: bool, capacity_lora_r: int):\n'
PGD_NEW_SIG = 'def build_backbone(checkpoint_path: str, r: int, alpha: int, include_ffn: bool, capacity_lora_r: int,\n                    msgproc_lora_r: int = None):\n'

PGD_OLD_BUILD = (
    '    backbone = VoiceMarkBackbone()\n'
    '    if include_ffn:\n'
    '        apply_lora_adapters(\n'
    '            backbone, r=r, alpha=alpha, targets=("msg_processor", "detector"),\n'
    '            include_ffn=True, ffn_r=capacity_lora_r, ffn_targets=("msg_processor",),\n'
    '        )\n'
    '    else:\n'
    '        apply_lora_adapters(backbone, r=r, alpha=alpha)\n'
)
PGD_NEW_BUILD = (
    '    backbone = VoiceMarkBackbone()\n'
    '    if include_ffn:\n'
    '        apply_lora_adapters(\n'
    '            backbone, r=r, alpha=alpha, targets=("msg_processor", "detector"),\n'
    '            include_ffn=True, ffn_r=capacity_lora_r, ffn_targets=("msg_processor",),\n'
    '        )\n'
    '    else:\n'
    '        # 2026-09-14 quality-tradeoff patch: msgproc_lora_r reconstructs a\n'
    '        # checkpoint trained with train_route2_clone_aware.py --msgproc_lora_r\n'
    '        # (independent, lower rank for msg_processor only) -- MUST match the\n'
    '        # value used at training time or load_state_dict will hit a shape\n'
    '        # mismatch on msg_processor\'s LoRA tensors.\n'
    '        target_ranks = {"msg_processor": msgproc_lora_r} if msgproc_lora_r else None\n'
    '        apply_lora_adapters(backbone, r=r, alpha=alpha, target_ranks=target_ranks)\n'
)

_apply(PGD_PATH, "msgproc_lora_r reconstructs a", [
    (PGD_OLD_SIG, PGD_NEW_SIG),
    (PGD_OLD_BUILD, PGD_NEW_BUILD),
])


# ---------------------------------------------------- train_route2_clone_aware.py
TRAIN_PATH = "src/train_route2_clone_aware.py"

TRAIN_OLD_FN = (
    'def build_trainable_backbone(checkpoint_path, lora_r, lora_alpha, train_msg_processor):\n'
    '    """\n'
    '    Loads the Stage-1 checkpoint, then unfreezes ONLY what this run trains.\n'
    '    Everything else -- codec, base weights, and (by default) msg_processor --\n'
    '    stays frozen, exactly as every other script in this project assumes.\n'
    '    """\n'
    '    backbone = VoiceMarkBackbone()\n'
    '    apply_lora_adapters(backbone, r=lora_r, alpha=lora_alpha)\n'
)
TRAIN_NEW_FN = (
    'def build_trainable_backbone(checkpoint_path, lora_r, lora_alpha, train_msg_processor, msgproc_lora_r=None):\n'
    '    """\n'
    '    Loads the Stage-1 checkpoint, then unfreezes ONLY what this run trains.\n'
    '    Everything else -- codec, base weights, and (by default) msg_processor --\n'
    '    stays frozen, exactly as every other script in this project assumes.\n'
    '\n'
    '    msgproc_lora_r (2026-09-14, quality-tradeoff patch): optional lower rank\n'
    '    applied ONLY to msg_processor\'s LoRA adapters, independent of detector\'s\n'
    '    rank (stays at lora_r). Motivated by the finding that neither --epochs nor\n'
    '    --lambda_clone changed --train_msg_processor\'s PESQ/SIM cost at all (both\n'
    '    landed statistically identical to the untouched run) -- pointing at LoRA\n'
    '    capacity itself, not training duration or loss weighting, as the lever.\n'
    '    """\n'
    '    backbone = VoiceMarkBackbone()\n'
    '    target_ranks = {"msg_processor": msgproc_lora_r} if msgproc_lora_r else None\n'
    '    apply_lora_adapters(backbone, r=lora_r, alpha=lora_alpha, target_ranks=target_ranks)\n'
)

TRAIN_OLD_ARGS = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    args = p.parse_args()\n'
)
TRAIN_NEW_ARGS = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    p.add_argument("--msgproc_lora_r", type=int, default=None,\n'
    '                    help="Lower LoRA rank for msg_processor ONLY (detector stays at "\n'
    '                         "--lora_r). Quality-tradeoff test: --epochs and --lambda_clone "\n'
    '                         "both failed to change --train_msg_processor\'s PESQ/SIM cost at "\n'
    '                         "all, suggesting the cost comes from LoRA capacity itself, not "\n'
    '                         "training duration or loss weighting. Only takes effect with "\n'
    '                         "--train_msg_processor. Try e.g. 2 or 4 (full rank default is 8). "\n'
    '                         "MUST be passed again, matching, to any eval script loading the "\n'
    '                         "resulting checkpoint (e.g. xtts_transfer_eval.py --msgproc_lora_r).")\n'
    '    args = p.parse_args()\n'
)

TRAIN_OLD_CALL = (
    '    backbone, trainable = build_trainable_backbone(\n'
    '        args.checkpoint, args.lora_r, args.lora_alpha, args.train_msg_processor)\n'
)
TRAIN_NEW_CALL = (
    '    backbone, trainable = build_trainable_backbone(\n'
    '        args.checkpoint, args.lora_r, args.lora_alpha, args.train_msg_processor,\n'
    '        msgproc_lora_r=args.msgproc_lora_r)\n'
)

_apply(TRAIN_PATH, "quality-tradeoff patch): optional lower rank", [
    (TRAIN_OLD_FN, TRAIN_NEW_FN),
    (TRAIN_OLD_ARGS, TRAIN_NEW_ARGS),
    (TRAIN_OLD_CALL, TRAIN_NEW_CALL),
])


# ------------------------------------------------------- xtts_transfer_eval.py
XTTS_PATH = "src/eval/xtts_transfer_eval.py"

XTTS_OLD_ARGS = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    p.add_argument("--epsilon", type=float, default=0.0)\n'
)
XTTS_NEW_ARGS = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    p.add_argument("--msgproc_lora_r", type=int, default=None,\n'
    '                    help="Only needed to load a checkpoint trained with "\n'
    '                         "train_route2_clone_aware.py --msgproc_lora_r -- MUST match "\n'
    '                         "the value used at training time or load_state_dict will hit "\n'
    '                         "a shape mismatch on msg_processor\'s LoRA tensors.")\n'
    '    p.add_argument("--epsilon", type=float, default=0.0)\n'
)

XTTS_OLD_CALL = (
    '    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,\n'
    '                               include_ffn=False, capacity_lora_r=32)\n'
)
XTTS_NEW_CALL = (
    '    backbone = build_backbone(args.checkpoint, args.lora_r, args.lora_alpha,\n'
    '                               include_ffn=False, capacity_lora_r=32,\n'
    '                               msgproc_lora_r=args.msgproc_lora_r)\n'
)

_apply(XTTS_PATH, "Only needed to load a checkpoint trained with", [
    (XTTS_OLD_ARGS, XTTS_NEW_ARGS),
    (XTTS_OLD_CALL, XTTS_NEW_CALL),
])


# ------------------------------------------------------ save_audio_samples.py
SAVE_PATH = "src/eval/save_audio_samples.py"

SAVE_OLD_FN = (
    'def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16):\n'
    '    backbone = VoiceMarkBackbone()\n'
    '    apply_lora_adapters(backbone, r=r, alpha=alpha)\n'
)
SAVE_NEW_FN = (
    'def build_backbone(lora_checkpoint_path: str = None, r: int = 8, alpha: int = 16, msgproc_lora_r: int = None):\n'
    '    backbone = VoiceMarkBackbone()\n'
    '    target_ranks = {"msg_processor": msgproc_lora_r} if msgproc_lora_r else None\n'
    '    apply_lora_adapters(backbone, r=r, alpha=alpha, target_ranks=target_ranks)\n'
)

SAVE_OLD_ARGS = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    args = p.parse_args()\n'
)
SAVE_NEW_ARGS = (
    '    p.add_argument("--lora_r", type=int, default=8)\n'
    '    p.add_argument("--lora_alpha", type=int, default=16)\n'
    '    p.add_argument("--msgproc_lora_r", type=int, default=None,\n'
    '                    help="Must match --msgproc_lora_r used at training time, if any.")\n'
    '    args = p.parse_args()\n'
)

SAVE_OLD_CALL = '    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha)\n'
SAVE_NEW_CALL = (
    '    backbone = build_backbone(lora_checkpoint_path=args.checkpoint, r=args.lora_r, alpha=args.lora_alpha,\n'
    '                               msgproc_lora_r=args.msgproc_lora_r)\n'
)

_apply(SAVE_PATH, "Must match --msgproc_lora_r used at training time", [
    (SAVE_OLD_FN, SAVE_NEW_FN),
    (SAVE_OLD_ARGS, SAVE_NEW_ARGS),
    (SAVE_OLD_CALL, SAVE_NEW_CALL),
])
