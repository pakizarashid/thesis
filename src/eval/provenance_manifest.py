"""
src/eval/provenance_manifest.py

Writes a manifest that ties every number to exact inputs (2026-10-24):

  - git commit, branch and uncommitted files
  - Python, package versions, GPU
  - SHA-256 and size of every file matching --files (checkpoints, the VoiceMark
    weights, the SpeechTokenizer weights and config, text pools, ...)
  - the evaluation and training clip IDs of the LibriSpeech subset (and VCTK if
    given), with the crop length used
  - free-form settings (--settings_json), e.g. the split arguments, clone texts,
    the Route 2 arms

Run it at the start of a session (inputs) and again at the end (all
checkpoints produced). Each run writes a new file; nothing is overwritten.

Usage:
    python src/eval/provenance_manifest.py --output results/manifest_vmaug_start.json \\
        --files "external/voicemark/voicemark.pth" "checkpoints/**/*.pt" \\
        --data_root /tmp/data/librispeech --settings_json '{"crop_seconds": 3}'
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "data"))

PACKAGES = ("torch", "torchaudio", "transformers", "tokenizers", "huggingface-hub", "coqui-tts", "f5-tts",
            "speechtokenizer", "encodec", "numpy", "scipy", "pesq", "pystoi", "faster-whisper", "speechbrain",
            "soundfile", "librosa", "omegaconf", "safetensors")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"error: {e}"


def librispeech_ids(args):
    from librispeech import LibriSpeechSubset
    out = {}
    for split in ("train", "eval"):
        ds = LibriSpeechSubset(root=args.data_root, n_speakers=args.n_speakers,
                               utterances_per_speaker=args.utterances_per_speaker,
                               n_eval_speakers=args.n_eval_speakers,
                               eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                               crop_seconds=args.crop_seconds, split=split)
        out[split] = [ds._full_dataset._walker[i] for i in ds._subset_indices]
    out["index_file_sha256"] = sha256(ds.index_cache_path) if os.path.exists(ds.index_cache_path) else None
    return out


def hf_space_check(space, files):
    """Compares local external/voicemark/* hashes with the LFS sha256 the Space reports.
    A match means the evaluated checkpoint is byte-identical to the authors' released file."""
    out = {"space": space}
    try:
        from huggingface_hub import HfApi
        info = HfApi().space_info(space, files_metadata=True)
        remote = {s.rfilename: (s.lfs.sha256 if getattr(s, "lfs", None) else None) for s in info.siblings}
        out["space_revision"] = info.sha
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    rows = []
    for f in files:
        if not f["path"].startswith("external/voicemark/"):
            continue
        rel = f["path"][len("external/voicemark/"):]
        r = remote.get(rel)
        rows.append({"file": rel, "local_sha256": f["sha256"], "space_lfs_sha256": r,
                     "match": (r == f["sha256"]) if r else None})
    out["files"] = rows
    out["all_lfs_files_match"] = all(r["match"] for r in rows if r["match"] is not None) and any(
        r["match"] is not None for r in rows)
    return out


def vctk_ids(args):
    from vctk import VCTKSubset
    ds = VCTKSubset(root=args.vctk_root, n_speakers=args.vctk_n_speakers,
                    utterances_per_speaker=args.vctk_utterances_per_speaker,
                    n_eval_speakers=args.n_eval_speakers,
                    eval_utterances_per_speaker=args.eval_utterances_per_speaker,
                    crop_seconds=args.crop_seconds, split="eval")
    return {"eval": [f"{it['speaker_id']}/{it['utterance_id']}" for it in ds._items]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--files", nargs="*", default=[], help="Glob patterns relative to the repo root.")
    p.add_argument("--settings_json", default="{}")
    p.add_argument("--data_root", default=None)
    p.add_argument("--vctk_root", default=None)
    p.add_argument("--n_speakers", type=int, default=60)
    p.add_argument("--utterances_per_speaker", type=int, default=15)
    p.add_argument("--n_eval_speakers", type=int, default=20)
    p.add_argument("--eval_utterances_per_speaker", type=int, default=5)
    p.add_argument("--vctk_n_speakers", type=int, default=10)
    p.add_argument("--vctk_utterances_per_speaker", type=int, default=10)
    p.add_argument("--crop_seconds", type=float, default=3.0)
    p.add_argument("--check_hf_space", default=None,
                   help="e.g. haiyunli/VoiceMark: compare the SHA-256 of local files under "
                        "external/voicemark with the LFS hashes the authors' Space publishes.")
    args = p.parse_args()

    from importlib.metadata import version, PackageNotFoundError
    pkgs = {}
    for name in PACKAGES:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pass

    files = []
    for pat in args.files:
        for f in sorted(glob.glob(os.path.join(REPO, pat), recursive=True)):
            if os.path.isfile(f):
                files.append({"path": os.path.relpath(f, REPO), "bytes": os.path.getsize(f), "sha256": sha256(f)})

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"commit": run("git rev-parse HEAD"), "branch": run("git rev-parse --abbrev-ref HEAD"),
                "log": run("git log --oneline -8").splitlines(),
                "uncommitted": [l for l in run("git status --porcelain").splitlines() if l][:200]},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").splitlines(),
        "packages": pkgs,
        "env": {k: os.environ.get(k) for k in ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "COQUI_TOS_AGREED")},
        "files": files,
        "settings": json.loads(args.settings_json),
        "crop_seconds": args.crop_seconds,
    }
    if args.check_hf_space:
        manifest["hf_space_check"] = hf_space_check(args.check_hf_space, files)
    if args.data_root:
        manifest["librispeech_clip_ids"] = librispeech_ids(args)
    if args.vctk_root and os.path.isdir(args.vctk_root):
        manifest["vctk_clip_ids"] = vctk_ids(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists; manifests are never overwritten")
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {len(files)} files hashed, git {manifest['git']['commit'][:10]}, -> {args.output}")


if __name__ == "__main__":
    main()
