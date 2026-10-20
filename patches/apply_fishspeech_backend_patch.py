"""
apply_fishspeech_backend_patch.py (2026-10-19)

Adds FishSpeech (v1.5) as a backend of src/eval/cloner_watermark_eval.py (--cloner fishspeech, --fish_url).

WHY A SERVER. FishSpeech pins its own torch / numpy / transformers stack. Notebooks 12 and 17 showed that every cloner needs its own environment, and
the detector, ECAPA, Resemblyzer and WER scorers live in the standard one. So the Fish model runs as its OWN process (tools/api_server.py, in its own
uv venv) and this backend only POSTs to it: nothing from fish_speech is imported here, the standard environment is untouched.

CLONE CALL. POST {url}/v1/tts, body = msgpack of {"text", "references": [{"audio": <wav bytes>, "text": <prompt transcript>}], "format": "wav"}.
Only these three fields are sent: every sampling setting stays at the server default (top_p 0.7, temperature 0.7, repetition penalty 1.2), no seed is
fixed (as for the other cloners). The prompt transcript comes from --auto_transcribe (faster-whisper), the same source as MaskGCT.

Requires in the calling environment:  pip install ormsgpack requests   (pure wheels; do not touch numpy/torch).
Usage: python patches/apply_fishspeech_backend_patch.py [path-to-cloner_watermark_eval.py]   (idempotent; apply AFTER apply_xtts_backend_patch.py)
"""
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "src/eval/cloner_watermark_eval.py"
src = open(TARGET).read()
orig = src


def sub(old, new, label):
    global src
    if src.count(old) != 1:
        raise SystemExit(f"[fish-backend patch] cannot apply '{label}': expected exactly 1 match, found {src.count(old)} in {TARGET}")
    src = src.replace(old, new)
    print(f"[fish-backend patch] applied: {label}")


FISH_BLOCK = '''# --------------------------------------------------------------------------------
# backend: FishSpeech v1.5  (language model over codec tokens of the reference; served by tools/api_server.py in its own environment)
# --------------------------------------------------------------------------------

def load_fishspeech(device, args):
    import time
    import requests
    url = args.fish_url.rstrip("/")
    last = None
    for _ in range(3):
        for method in ("get", "post"):
            try:
                r = getattr(requests, method)(url + "/v1/health", timeout=15)
                if r.status_code == 200:
                    print(f"[fishspeech] server answers at {url} ({method.upper()} /v1/health: {r.text[:80]!r})")
                    return {"url": url}
                last = f"HTTP {r.status_code}"
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
        time.sleep(5)
    raise SystemExit(f"FishSpeech server not reachable at {url} ({last}). Start tools/api_server.py first (see notebook 18, section 4).")


def clone_fishspeech(model, speaker_audio_16k, gen_text, tmp_dir, tag, ref_text, args):
    import io
    import requests
    import ormsgpack
    import soundfile as sf
    ref_path = write_ref_wav(speaker_audio_16k, tmp_dir, tag)
    if not ref_text:
        raise SystemExit("FishSpeech needs the prompt transcript; pass --auto_transcribe.")
    with open(ref_path, "rb") as fh:
        ref_bytes = fh.read()
    payload = {"text": gen_text, "references": [{"audio": ref_bytes, "text": ref_text}], "format": "wav"}
    r = requests.post(model["url"] + "/v1/tts", data=ormsgpack.packb(payload),
                      headers={"content-type": "application/msgpack"}, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"FishSpeech server answered HTTP {r.status_code}: {r.text[:300]}")
    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < sr // 2:
        raise RuntimeError(f"FishSpeech returned {len(wav) / sr:.2f} s of audio")
    return to_16k_tensor(wav, sr, speaker_audio_16k.device)


'''

if "def load_fishspeech" not in src:
    sub("BACKENDS = {\n", FISH_BLOCK + "BACKENDS = {\n" +
        '    "fishspeech": (load_fishspeech, clone_fishspeech, "LM over codec tokens of the reference", None),\n', "fishspeech backend + BACKENDS entry")
else:
    print("[fish-backend patch] backend already present")

if "--fish_url" not in src:
    sub('    p.add_argument("--amphion_root", type=str, default=None)\n',
        '    p.add_argument("--amphion_root", type=str, default=None)\n'
        '    p.add_argument("--fish_url", type=str, default="http://127.0.0.1:8080", help="FishSpeech only: address of tools/api_server.py")\n', "--fish_url argument")
if 'args.cloner in ("cosyvoice", "maskgct") and not' in src:
    sub('args.cloner in ("cosyvoice", "maskgct") and not', 'args.cloner in ("cosyvoice", "maskgct", "fishspeech") and not', "transcript requirement")

if src != orig:
    open(TARGET, "w").write(src)
    print(f"[fish-backend patch] wrote {TARGET}")
else:
    print("[fish-backend patch] nothing to do")
