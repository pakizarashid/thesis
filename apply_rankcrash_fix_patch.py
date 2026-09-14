"""
apply_rankcrash_fix_patch.py -- fixes the shape-mismatch crash seen when
running train_route2_clone_aware.py --train_msg_processor --msgproc_lora_r 2
against ./checkpoints/stage1_scaleup_aug/stage1_epoch9.pt.

Root cause: stage1_epoch9.pt's saved lora_state_dict includes msg_processor's
LoRA tensors at rank 8 (that's what Stage 1 trained, uniformly, before this
rank-override feature existed). build_trainable_backbone() now builds
msg_processor's adapters at msgproc_lora_r=2, then unconditionally does
`backbone.model.load_state_dict(ckpt["lora_state_dict"], strict=False)`.
strict=False only tolerates missing/extra key NAMES -- it does NOT tolerate a
shape mismatch on a key that exists on both sides, so this raises instead of
skipping. There is no clean way to "truncate" an existing rank-8 LoRA A/B pair
down to rank 2 (they're independent random-init low-rank factors, not an SVD
of a single matrix), so the correct, honest fix is: msg_processor's adapters
start from LoRA zero-init at the new rank (same starting point any first
--train_msg_processor run has before msg_processor is ever trained) while the
detector's rank-8 tensors (unaffected, still matching) load normally as
before. This is applied loudly -- every skipped tensor is printed with both
shapes -- not silently, so it's visible in the training log.

Idempotent: safe to re-run (checked via a marker string).

Usage: python3 apply_rankcrash_fix_patch.py <repo-root-or-.>
"""
import os
import sys

MARKER = "msgproc_lora_r patch: the init checkpoint's LoRA tensors"


def _apply(path, edits):
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} not found (cwd={os.getcwd()}).")
    content = open(path, "r").read()
    if MARKER in content:
        print(f"[apply_rankcrash_fix_patch] {path} already patched -- skipping.")
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
    print(f"[apply_rankcrash_fix_patch] Patched {path} ({len(edits)} edits applied).")


TRAIN_PATH = "src/train_route2_clone_aware.py"

OLD = (
    "    if checkpoint_path is not None:\n"
    "        ckpt = torch.load(checkpoint_path, map_location=\"cpu\", weights_only=False)\n"
    "        backbone.model.load_state_dict(ckpt[\"lora_state_dict\"], strict=False)\n"
    "        print(f\"[build] Loaded LoRA weights from {checkpoint_path} \"\n"
    "              f\"(epoch {ckpt.get('epoch')}, train-time avg_acc={ckpt.get('avg_acc')})\")\n"
)
NEW = (
    "    if checkpoint_path is not None:\n"
    "        ckpt = torch.load(checkpoint_path, map_location=\"cpu\", weights_only=False)\n"
    "        state_dict = ckpt[\"lora_state_dict\"]\n"
    "        # 2026-09-14 msgproc_lora_r patch: the init checkpoint's LoRA tensors\n"
    "        # were saved at their ORIGINAL rank (e.g. Stage 1's msg_processor LoRA\n"
    "        # is rank 8). If msgproc_lora_r asks for a different rank, those specific\n"
    "        # tensors can't be loaded -- LoRA A/B factors at different ranks aren't\n"
    "        # reshapeable into each other (no clean truncation), so msg_processor's\n"
    "        # adapters fall back to LoRA zero-init instead (the same starting point\n"
    "        # any first --train_msg_processor run has before msg_processor is ever\n"
    "        # trained). strict=False alone does NOT handle this -- it only tolerates\n"
    "        # missing/extra key NAMES, not a shape mismatch on a key present in both\n"
    "        # -- so mismatched keys are dropped explicitly first, loudly, before\n"
    "        # load_state_dict runs.\n"
    "        own_state = backbone.model.state_dict()\n"
    "        skipped = [k for k in state_dict\n"
    "                   if k in own_state and state_dict[k].shape != own_state[k].shape]\n"
    "        if skipped:\n"
    "            print(f\"[build] msgproc_lora_r={msgproc_lora_r}: {len(skipped)} checkpoint \"\n"
    "                  f\"tensor(s) don't match the new rank and will NOT be warm-started \"\n"
    "                  f\"(LoRA zero-init instead):\")\n"
    "            for k in skipped:\n"
    "                print(f\"    {k}: checkpoint {tuple(state_dict[k].shape)} \"\n"
    "                      f\"vs model {tuple(own_state[k].shape)}\")\n"
    "            state_dict = {k: v for k, v in state_dict.items() if k not in skipped}\n"
    "        backbone.model.load_state_dict(state_dict, strict=False)\n"
    "        print(f\"[build] Loaded LoRA weights from {checkpoint_path} \"\n"
    "              f\"(epoch {ckpt.get('epoch')}, train-time avg_acc={ckpt.get('avg_acc')})\")\n"
)

_apply(TRAIN_PATH, [(OLD, NEW)])
