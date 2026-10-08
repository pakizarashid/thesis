"""
plot_audio_comparisons.py  (v2, 2026-10-08)  --  LISTEN to and LOOK AT the audio at every epsilon, side by side.

What it makes (all in --out):
  gallery.html            one page, no server needed: for each chosen utterance a table with
                            original | watermark-only | protected @ each eps | clone of each   (audio players + spectrograms)
  sample{i}_spectrograms.png   the same, as pictures (3 columns: what a listener hears / what was ADDED / what the attacker's clone looks like)
  added_signal_spectrum.png    average spectrum of (protected - original) per eps vs the original: WHERE the perturbation sits
  blind_test.html         optional (--blind): a "which one is the processed clip?" listening test for row 1e of the success threshold
  wav/                    every clip as a plain WAV, in case you want to open them elsewhere

Where it runs: in the SAME Kaggle session that ran notebook 01 sections 7 / 9b. The protected WAVs and the clones are NOT committed to
git (only the result JSONs are), so a fresh session has to regenerate them first (notebook 01, RUN["eps_sweep"] / RUN["three_audio"]).

Typical use (Kaggle cell, after notebook 01 has run its set-up cell, so REPO and COMPOSABILITY_DIR exist):

    !python {REPO}/src/eval/plot_audio_comparisons.py --root {REPO} --composability_dir {COMPOSABILITY_DIR} \
        --eps 0.002 0.01 0.02 0.04 0.08 --samples 0 3 7 12 --suffix _b --out /kaggle/working/eps_gallery --blind

then download /kaggle/working/eps_gallery (right-click -> download in the Output tab) and open gallery.html.

Needs only numpy, scipy, matplotlib (all on Kaggle). No librosa / soundfile.

Which file is which (this is how the pipeline names them):
  protected audio  : <eps dir>/sample{i}_reference.wav   TRUE clean original
                     <eps dir>/sample{i}_unprotected.wav watermark only (eps = 0)       [composability dir only]
                     <eps dir>/sample{i}_protected.wav   watermark + PGD at that eps
                     <eps dir> = --composability_dir for eps 0.002, else <root>/eps_sweep_route2/eps{eps}_audio
  clones (F5-TTS)  : <root>/eps_sweep_route2/clones_route2_epssweep_eps{eps}{suffix}/sample{i}_clone_f5tts.wav
                     <root>/eps_sweep_route2/clones_base_unprot_b/sample{i}_clone_f5tts.wav     (clone of the watermark-only audio)
"""
import argparse, base64, io, json, os, random, sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000


# ----------------------------------------------------------------------------- io
def read_wav(path):
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"{path} is {sr} Hz, expected {SR}")
    return x


def wav_bytes(x):
    buf = io.BytesIO()
    wavfile.write(buf, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


def b64(b):
    return base64.b64encode(b).decode()


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def sim_by_index(cond):
    """ecapa_sim_eval.py writes values in sorted-FILENAME order (sample0, sample10 ... sample1 ...); return them in numeric order."""
    vals = cond["values"]
    idx = cond.get("indices") or sorted(range(len(vals)), key=lambda i: f"sample{i}_reference.wav")
    out = [None] * len(vals)
    for v, i in zip(vals, idx):
        out[i] = v
    return out


# ----------------------------------------------------------------------------- locate files
class Layout:
    def __init__(self, a):
        self.root, self.comp, self.suffix = a.root, a.composability_dir, a.suffix

    def adir(self, eps):
        cands = []
        if abs(float(eps) - 0.002) < 1e-12 and self.comp:
            cands.append(self.comp)
        cands.append(os.path.join(self.root, "eps_sweep_route2", f"eps{eps}_audio"))
        for d in cands:
            if os.path.exists(os.path.join(d, "sample0_protected.wav")):
                return d
        return None

    def tag(self, eps):
        for t in (f"route2_epssweep_eps{eps}{self.suffix}", f"route2_epssweep_eps{eps}"):
            if os.path.isdir(os.path.join(self.root, "eps_sweep_route2", f"clones_{t}")):
                return t
        return f"route2_epssweep_eps{eps}{self.suffix}"

    def clone(self, eps, i):
        p = os.path.join(self.root, "eps_sweep_route2", f"clones_{self.tag(eps)}", f"sample{i}_clone_f5tts.wav")
        return p if os.path.exists(p) else None

    def base_clone(self, i):
        for t in ("base_unprot_b", "base_unprot"):
            p = os.path.join(self.root, "eps_sweep_route2", f"clones_{t}", f"sample{i}_clone_f5tts.wav")
            if os.path.exists(p):
                return p
        return None

    def metrics(self, tag, i):
        """(SIM, ACC) of clip i for a tag, or (None, None)."""
        e = jload(os.path.join(self.root, "results", f"results_ecapa_sim_{tag}.json"))
        a = jload(os.path.join(self.root, "results", f"results_{tag}_acc_protected.json"))
        sim = acc = None
        try:
            sim = sim_by_index(e["results"]["clone_f5tts"])[i]
        except Exception:
            pass
        try:
            acc = a["results"]["acc_clone_values"][i]
        except Exception:
            pass
        return sim, acc


# ----------------------------------------------------------------------------- spectrograms
def spec_db(x, ref_peak):
    f, t, S = spectrogram(x, fs=SR, nperseg=512, noverlap=384, window="hann", mode="magnitude")
    return f, t, 20 * np.log10(S / ref_peak + 1e-9)


def draw(ax, x, ref_peak, title, vmin=-90, cmap="magma"):
    f, t, D = spec_db(x, ref_peak)
    ax.pcolormesh(t, f / 1000, D, vmin=vmin, vmax=0, cmap=cmap, shading="auto", rasterized=True)
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("kHz", fontsize=7)
    ax.tick_params(labelsize=7)


def sample_figure(i, rows, out_png):
    """rows: list of dict(label, heard, added, clone, note). heard/added/clone are arrays or None."""
    ref = rows[0]["heard"]
    peak = float(np.max(spectrogram(ref, fs=SR, nperseg=512, noverlap=384, mode="magnitude")[2])) + 1e-9
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.2 * n), squeeze=False)
    for r, row in enumerate(rows):
        for c, key, cap in ((0, "heard", "heard"), (1, "added", "ADDED (this - original)"), (2, "clone", "attacker's clone")):
            ax = axes[r, c]
            x = row.get(key)
            if x is None:
                ax.axis("off")
                continue
            draw(ax, x, peak, f"{row['label']} | {cap}")
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=7)
    fig.suptitle(f"sample {i}   (same colour scale in every panel: 0 dB = loudest bin of the original; -90 dB = black)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def added_spectrum(per_eps, orig_list, out_png):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    def mean_spec(xs):
        acc = None
        for x in xs:
            f, _, S = spectrogram(x, fs=SR, nperseg=512, noverlap=384, mode="magnitude")
            m = S.mean(axis=1)
            acc = m if acc is None else acc + m
        return f, 20 * np.log10(acc / len(xs) + 1e-9)

    f, o = mean_spec(orig_list)
    ax.plot(f / 1000, o, color="black", lw=2, label="original (speech)")
    for eps, deltas in per_eps.items():
        f, d = mean_spec(deltas)
        ax.plot(f / 1000, d, lw=1.3, label=f"added, eps {eps}")
    ax.set_xlabel("kHz")
    ax.set_ylabel("average magnitude (dB)")
    ax.set_title("Where the protection puts its energy (protected - original), averaged over the chosen samples\n"
                 "a line close to the black one = the added noise is as loud as the speech in that band", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- html
CSS = """body{font-family:system-ui,Arial,sans-serif;margin:16px;max-width:1200px;background:#fff;color:#111}
table{border-collapse:collapse;width:100%;margin:10px 0 28px}th,td{border:1px solid #ccc;padding:6px 8px;font-size:13px;vertical-align:middle}
th{background:#f2f2f2;text-align:left}audio{width:230px;height:32px}img{max-width:100%}.n{color:#555;font-size:12px}
h2{margin-top:34px}"""


def audio_tag(x):
    if x is None:
        return "<span class=n>n/a</span>"
    return f'<audio controls preload="none" src="data:audio/wav;base64,{b64(wav_bytes(x))}"></audio>'


def build_gallery(sections, out_html, notes):
    parts = [f"<!doctype html><meta charset=utf-8><title>eps gallery</title><style>{CSS}</style>",
             "<h1>Listen / look: original vs protected vs clone, per epsilon</h1>", f"<p class=n>{notes}</p>"]
    for s in sections:
        parts.append(f"<h2>sample {s['i']}</h2>")
        if s.get("png"):
            parts.append(f"<img src='data:image/png;base64,{b64(open(s['png'], 'rb').read())}'>")
        parts.append("<table><tr><th>condition</th><th>what a listener hears</th><th>the attacker's clone</th><th>numbers for this clip</th></tr>")
        for r in s["rows"]:
            parts.append(f"<tr><td>{r['label']}</td><td>{audio_tag(r['heard'])}</td><td>{audio_tag(r.get('clone'))}</td><td class=n>{r.get('note', '')}</td></tr>")
        parts.append("</table>")
    with open(out_html, "w") as f:
        f.write("\n".join(parts))


def build_blind(trials, out_html):
    """Each trial: two clips A/B, one is the original, one is protected. 'Which is the processed one?' -> % correct per eps."""
    data = [dict(a=b64(wav_bytes(t["A"])), b=b64(wav_bytes(t["B"])), key=t["key"], eps=t["eps"]) for t in trials]
    html = """<!doctype html><meta charset=utf-8><title>blind test</title><style>__CSS__ .t{border:1px solid #ccc;padding:8px;margin:8px 0}</style>
<h1>Blind listening check (row 1e)</h1>
<p>For each trial, one clip is the untouched original and the other is the PROTECTED version. Choose the one you think is processed
(a guess is fine). Headphones, same volume, do not look at the page source. Press Score at the end.</p>
<p class=n>Reading the result: 50% = you cannot tell them apart. The success threshold says 'inaudible' only if correct is at most 60% -- but with only
~10 trials per epsilon that cannot be told from luck (10 trials: 6 or more correct happens by chance ~38% of the time), so use many trials and several listeners.</p>
<div id=root></div><button onclick=score() style="font-size:16px;padding:8px 18px">Score</button><pre id=out></pre>
<script>
const T=__DATA__;
const root=document.getElementById('root');
T.forEach((t,i)=>{const d=document.createElement('div');d.className='t';
d.innerHTML='<b>Trial '+(i+1)+'</b><br>A <audio controls preload=none src="data:audio/wav;base64,'+t.a+'"></audio>  B <audio controls preload=none src="data:audio/wav;base64,'+t.b+'"></audio><br>'+
'Processed one is: <label><input type=radio name=q'+i+' value=A> A</label> <label><input type=radio name=q'+i+' value=B> B</label>';root.appendChild(d);});
function score(){const per={};let n=0;
T.forEach((t,i)=>{const v=document.querySelector('input[name=q'+i+']:checked');if(!v)return;n++;per[t.eps]=per[t.eps]||[0,0];per[t.eps][1]++;if(v.value===t.key)per[t.eps][0]++;});
let s='answered '+n+' of '+T.length+'\\n';for(const e in per)s+='eps '+e+': '+per[e][0]+'/'+per[e][1]+' correct = '+Math.round(100*per[e][0]/per[e][1])+'%\\n';
document.getElementById('out').textContent=s;}
</script>""".replace("__CSS__", CSS).replace("__DATA__", json.dumps(data))
    with open(out_html, "w") as f:
        f.write(html)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="the cloned repo, e.g. /kaggle/working/thesis")
    p.add_argument("--composability_dir", default=None, help="folder with sample{i}_reference/_unprotected/_protected.wav at eps 0.002")
    p.add_argument("--eps", nargs="+", default=["0.002", "0.01", "0.02", "0.04", "0.08"])
    p.add_argument("--samples", nargs="+", type=int, default=[0, 3, 7, 12])
    p.add_argument("--suffix", default="_b", help="tag suffix of the draw whose clones to show ('_b' = the section 9b draw, '' = the committed draw)")
    p.add_argument("--out", default="eps_gallery")
    p.add_argument("--match_loudness", action="store_true", help="scale every protected clip to the original's RMS before you listen (removes the loudness cue)")
    p.add_argument("--blind", action="store_true", help="also write blind_test.html")
    p.add_argument("--blind_per_eps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    L = Layout(a)
    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    eps_ok = [e for e in a.eps if L.adir(e)]
    for e in a.eps:
        if e not in eps_ok:
            print(f"[skip] eps {e}: no sample0_protected.wav in {os.path.join(a.root, 'eps_sweep_route2', f'eps{e}_audio')}"
                  f"{' or the composability dir' if a.composability_dir else ''} -- generate it first (notebook 01 section 7 / 9b)")
    if not eps_ok:
        sys.exit("nothing to show")

    sections, per_eps_delta, origs = [], {e: [] for e in eps_ok}, []
    for i in a.samples:
        d0 = L.adir(eps_ok[0])
        if not os.path.exists(os.path.join(d0, f"sample{i}_reference.wav")):
            print(f"[skip] sample {i}: not in {d0}")
            continue
        orig = read_wav(os.path.join(d0, f"sample{i}_reference.wav"))
        origs.append(orig)
        rows = [dict(label="ORIGINAL (clean)", heard=orig, added=None, clone=None, note="the reference every score is measured against")]
        wm_dir = a.composability_dir or d0
        wm_path = os.path.join(wm_dir, f"sample{i}_unprotected.wav")
        if os.path.exists(wm_path):
            wm = read_wav(wm_path)
            n = min(len(orig), len(wm))
            bc = L.base_clone(i)
            bsim, bacc = L.metrics("base_unprot_b", i)
            note = f"clone SIM {bsim:.2f}  watermark {bacc * 16:.0f}/16 bits" if bsim is not None and bacc is not None else ""
            rows.append(dict(label="watermark only (eps 0)", heard=wm, added=wm[:n] - orig[:n], clone=read_wav(bc) if bc else None, note=note))
        for e in eps_ok:
            d = L.adir(e)
            pr = os.path.join(d, f"sample{i}_protected.wav")
            if not os.path.exists(pr):
                continue
            raw = read_wav(pr)
            ref_e = read_wav(os.path.join(d, f"sample{i}_reference.wav"))
            n = min(len(raw), len(ref_e))
            delta = raw[:n] - ref_e[:n]
            per_eps_delta[e].append(delta)
            x = raw * (10 ** ((rms_db(ref_e) - rms_db(raw)) / 20)) if a.match_loudness else raw
            cp = L.clone(e, i)
            sim, acc = L.metrics(L.tag(e), i)
            note = ""
            if sim is not None:
                note += f"clone SIM {sim:.2f} ({'attacker succeeds' if sim > 0.25 else 'clone fails'})"
            if acc is not None:
                note += f" | watermark in clone {acc * 16:.0f}/16 bits ({'traced' if acc * 16 >= 13 - 1e-9 else 'NOT traced'})"
            note += f" | loudness {rms_db(raw) - rms_db(ref_e):+.1f} dB vs original"
            rows.append(dict(label=f"PROTECTED eps {e}", heard=x, added=delta, clone=read_wav(cp) if cp else None, note=note))
        png = os.path.join(a.out, f"sample{i}_spectrograms.png")
        sample_figure(i, rows, png)
        for r in rows:
            tag = r["label"].split("(")[0].strip().lower().replace(" ", "_")
            for k in ("heard", "clone"):
                if r.get(k) is not None:
                    wavfile.write(os.path.join(a.out, "wav", f"sample{i}_{tag}_{k}.wav"), SR, (np.clip(r[k], -1, 1) * 32767).astype(np.int16))
        sections.append(dict(i=i, rows=rows, png=png))
        print(f"sample {i}: {len(rows)} rows")

    if not sections:
        sys.exit("no samples found")
    added_spectrum({e: v for e, v in per_eps_delta.items() if v}, origs, os.path.join(a.out, "added_signal_spectrum.png"))
    notes = ("Each row: left player = the PROTECTED audio a normal listener would get; right player = what F5-TTS produced when the attacker cloned it. "
             "If the protection works the clone says the same words in a DIFFERENT voice. Spectrogram colour scale is identical in every panel. "
             + ("Protected clips were loudness-matched to the original. " if a.match_loudness else "Loudness is NOT matched (row 1d of the threshold allows +-1 dB). "))
    build_gallery(sections, os.path.join(a.out, "gallery.html"), notes)
    print("wrote", os.path.join(a.out, "gallery.html"))

    if a.blind:
        rnd = random.Random(a.seed)
        trials = []
        pool = [s["i"] for s in sections]
        for e in eps_ok:
            for k in range(a.blind_per_eps):
                i = pool[k % len(pool)]
                d = L.adir(e)
                o, x = read_wav(os.path.join(d, f"sample{i}_reference.wav")), read_wav(os.path.join(d, f"sample{i}_protected.wav"))
                n = min(len(o), len(x))
                o, x = o[:n], x[:n]
                if a.match_loudness:
                    x = x * (10 ** ((rms_db(o) - rms_db(x)) / 20))
                trials.append(dict(A=x, B=o, key="A", eps=e) if rnd.random() < 0.5 else dict(A=o, B=x, key="B", eps=e))
        rnd.shuffle(trials)
        build_blind(trials, os.path.join(a.out, "blind_test.html"))
        print("wrote", os.path.join(a.out, "blind_test.html"), f"({len(trials)} trials)")


if __name__ == "__main__":
    main()
