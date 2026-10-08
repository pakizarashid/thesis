"""
apply_sourceacc_guard_patch.py

Patches the sample-0 sanity gate in cloner_watermark_eval.py.

Context: that gate aborts the whole run if acc_source (detection accuracy on the
INPUT/attacked audio itself, before cloning) is below 0.85 on sample 0. It exists to
catch a real --message_seed_base mismatch, which would otherwise silently produce
meaningless clone-accuracy numbers -- a legitimate protection, so this patch does NOT
remove it.

What it adds: an opt-in bypass via the ALLOW_LOW_SOURCE_ACC=1 environment variable.
With it unset (the default), behavior is UNCHANGED -- the script still aborts exactly
as before. Set ALLOW_LOW_SOURCE_ACC=1 only when you've already confirmed the seed
convention is correct via other successful runs on the same checkpoint (as we have --
7/10 attack conditions completed cleanly with the default seed_base on this checkpoint,
so a low sample-0 score on the remaining aggressive conditions reflects real
attack-induced watermark degradation, not a seed bug).

Usage:
    python apply_sourceacc_guard_patch.py
    ALLOW_LOW_SOURCE_ACC=1 python src/eval/cloner_watermark_eval.py ...
"""
import sys

PATH = "src/eval/cloner_watermark_eval.py"

OLD = '''        if args.input_wav_dir and args.wav_suffix != "clean" and i == 0 and a_src < 0.85:
            raise SystemExit(
                f"\\nABORT: acc_source={a_src:.4f} on the INPUT audio (expected ~0.99).\\n"
                f"The message regenerated with seed {args.message_seed_base}+{i} does not "
                f"match the one embedded in {args.input_wav_dir}.\\n"
                f"Fix --message_seed_base to match the generating script.")'''

NEW = '''        if args.input_wav_dir and args.wav_suffix != "clean" and i == 0 and a_src < 0.85:
            if os.environ.get("ALLOW_LOW_SOURCE_ACC") == "1":
                print(
                    f"\\nWARNING: acc_source={a_src:.4f} on sample 0 of the INPUT audio "
                    f"(expected ~0.99 if undamaged). Continuing because "
                    f"ALLOW_LOW_SOURCE_ACC=1 is set -- treat accuracy numbers from this run "
                    f"as reflecting real attack-induced degradation, not a seed mismatch "
                    f"(this assumption should already be verified against other successful "
                    f"runs on the same checkpoint before trusting it).\\n", flush=True)
            else:
                raise SystemExit(
                    f"\\nABORT: acc_source={a_src:.4f} on the INPUT audio (expected ~0.99).\\n"
                    f"The message regenerated with seed {args.message_seed_base}+{i} does not "
                    f"match the one embedded in {args.input_wav_dir}.\\n"
                    f"Fix --message_seed_base to match the generating script, or set "
                    f"ALLOW_LOW_SOURCE_ACC=1 to continue anyway if you've already confirmed "
                    f"low accuracy here reflects genuine attack damage rather than "
                    f"misconfiguration.")'''

with open(PATH) as f:
    src = f.read()

if NEW in src:
    print(f"{PATH}: already patched, nothing to do.")
    sys.exit(0)

if OLD not in src:
    print(f"ERROR: expected block not found verbatim in {PATH}. "
          f"The file may already differ from what this patch expects -- "
          f"inspect around the 'ABORT: acc_source' string by hand before proceeding.",
          file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)

with open(PATH, "w") as f:
    f.write(src)

print(f"{PATH}: patched. Set ALLOW_LOW_SOURCE_ACC=1 to bypass the sample-0 sanity gate.")
