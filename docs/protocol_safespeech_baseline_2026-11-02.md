# Protocol B1 — SafeSpeech as a reproduced protection baseline

Written 2026-11-02, before any SafeSpeech output exists. This is the protection-side equivalent of the VoiceMark
reproduction: the thing that lets the thesis say "compared with" instead of "mentioned alongside".

## Why SafeSpeech and not AntiFake

AntiFake (CCS 2023) is the closest paper by threat model — zero-shot, per-clip, ensemble of speaker encoders —
and our A0 arm is structurally its recipe. But its code could not be located, and the ACM full text was not
reachable, so its datasets, encoders and budget are unverified. It stays a cited reference, not a reproduced
baseline. (`cyberrrange/de-antifake` evaluates against AntiFake and may carry an implementation; worth one check
before settling.)

SafeSpeech (USENIX Security 2025) has public code (`wxzyd123/SafeSpeech`, Zenodo archive, artifact appendix),
publishes zero-shot numbers on F5-TTS — one of our cloners — and uses an explicit budget we can match.

## What is run, and what is not

Their full pipeline fine-tunes BERT-VITS2 and needs an 80 GB GPU. **We do not run it.** We need only the
perturbation generator:

1. `protect.py` + `save_audio.py` on our 100 clips, their released settings (ε = 8/255, α = 0.05, β = 10,
   BERT-VITS2 surrogate).
2. Clone the output with our F5-TTS / MaskGCT / CosyVoice 2 / XTTS-v2 / YourTTS.
3. Score with our harness: Resemblyzer at the frozen 0.687 threshold, exact McNemar, speaker-bootstrap CIs.

## Two conditions, both required

This is the fairness point raised by the author on 2026-10-25 and it is not optional:

- **B1-clean** — SafeSpeech on unwatermarked audio. Their intended setting. This is the honest comparison of
  their method against ours.
- **B1-wm** — SafeSpeech on RQA-watermarked audio. Tests how their method behaves *inside* our combined system.

Reporting only B1-wm would understate their method (it never expected a watermark). Reporting only B1-clean would
dodge the question of whether another protection can coexist with our watermark. Both, always.

## Conditions that must be stated whenever a number is compared

Any comparison is invalid unless these are reported beside it:

| | ours (Q2) | SafeSpeech |
|---|---|---|
| peak perturbation | 0.015 | 8/255 = 0.031 |
| surrogate | F5-TTS | BERT-VITS2 |
| carries a watermark | yes | no |
| per-clip budget spent on message preservation | yes | none |
| threat model | zero-shot | fine-tuning (primary) and zero-shot |

Their published zero-shot figure — ECAPA SIM 0.885 → 0.094 on F5-TTS — is **not** comparable to our ECAPA
0.475 → 0.105. The clean baselines differ twofold, which means the enrolment protocol and prompt length differ.
Only numbers produced by our own harness on our own clips may be compared.

## What is reported

- Acceptance at 0.687 for SafeSpeech (both conditions), Q2, and watermark-only, on all five cloners.
- Audio quality: PESQ, STOI of each protected file on the same 100 clips.
- Traced rate for B1-wm (does our watermark survive their perturbation?).
- A budget-normalised note: SafeSpeech operates at twice our peak, so any advantage it shows is reported
  together with its quality cost rather than as a method advantage.

## What no result here licenses

- No claim that our method beats SafeSpeech, or the reverse, unless the budgets are matched; they are not, and
  matching them is a separate experiment.
- No claim about SafeSpeech's own fine-tuning results, which are not reproduced.
- A failure of SafeSpeech in our harness is evidence about transfer to our cloners and clips, not about the
  paper.

## Deliverables

- `results/anticlone/prompts/safespeech/{clean,wm}/`
- `results/ce_ac_*_safespeech_{clean,wm}.json`
- a comparison table with the conditions above printed beside every number
