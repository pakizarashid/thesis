"""
src/data/vctk.py

THIRD rewrite, after two previous approaches failed:
  1. torchaudio.datasets.VCTK_092 -- exhausted Kaggle disk (zip + extraction
     simultaneously exceeded the 19.5GB budget).
  2. HuggingFace streaming (CSTR-Edinburgh/vctk) -- streaming=True doesn't
     actually avoid a full download for datasets using a custom loading
     script, which this one does; downloaded the whole corpus anyway.

THIS version reads directly from a Kaggle "Input" dataset
(pratt3000/vctk-corpus, mounted read-only at /kaggle/input/...) -- no
download, no extraction, no streaming complexity. Kaggle Input datasets are
mounted separately from /kaggle/working's disk quota, so this costs nothing
against the working-directory budget regardless of the dataset's actual size.

Confirmed directory structure via direct `find` inspection (not assumed):
    <root>/wav48/<speaker_id>/<speaker_id>_<utterance_num>.wav
    <root>/txt/<speaker_id>/<speaker_id>_<utterance_num>.txt

Mirrors LibriSpeechSubset's public API (same constructor arguments, same
batch dict keys) so existing eval scripts can point at this loader with
minimal changes.
"""

import os
import json
import random
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class VCTKSubset(Dataset):
    def __init__(self, root: str = "/kaggle/input/datasets/pratt3000/vctk-corpus/VCTK-Corpus/VCTK-Corpus",
                 n_speakers: int = 10, utterances_per_speaker: int = 10, n_eval_speakers: int = 5,
                 eval_utterances_per_speaker: int = 5, sample_rate: int = 16000,
                 crop_seconds: float = 3.0, split: str = "train", seed: int = 42,
                 index_cache_dir: str = "./data/vctk_index"):
        self.root = root
        self.wav_dir = os.path.join(root, "wav48")
        self.txt_dir = os.path.join(root, "txt")
        self.sample_rate = sample_rate
        self.crop_seconds = crop_seconds
        self.crop_samples = int(crop_seconds * sample_rate)
        self.split = split
        self._native_sample_rate = 48000

        if not os.path.isdir(self.wav_dir):
            raise FileNotFoundError(
                f"[VCTKSubset] {self.wav_dir} not found -- confirm the dataset is attached "
                f"via Kaggle's '+ Add Input' panel and this path matches what 'find' showed."
            )

        # Index caching is metadata-only (which speaker/utterance IDs were
        # selected) -- cheap, tiny file, no audio ever cached to disk since
        # we read directly from the mounted (already-present) source every time.
        os.makedirs(index_cache_dir, exist_ok=True)
        index_path = os.path.join(index_cache_dir, "subset_index.json")
        params = {"n_speakers": n_speakers, "utterances_per_speaker": utterances_per_speaker,
                   "n_eval_speakers": n_eval_speakers, "eval_utterances_per_speaker": eval_utterances_per_speaker,
                   "seed": seed}

        if os.path.exists(index_path):
            with open(index_path) as f:
                cached = json.load(f)
            if cached.get("params") == params:
                print(f"[VCTKSubset] Loaded cached subset index from {index_path}")
                self._index = cached
            else:
                print("[VCTKSubset] Cached index params don't match requested -- rebuilding.")
                self._index = self._build_index(params, index_path)
        else:
            self._index = self._build_index(params, index_path)

        key = "train_items" if split == "train" else "eval_items"
        self._items = self._index[key]
        n_spk = len(set(item["speaker_id"] for item in self._items))
        print(f"[VCTKSubset] Using {len(self._items)} utterances ({split} split, "
              f"{n_spk} speakers, speaker-disjoint from the other split). "
              f"Reading directly from mounted Kaggle input -- zero working-disk cost.")

    def _build_index(self, params, index_path):
        print("[VCTKSubset] Building speaker index (one-time directory scan)...")
        speakers = sorted(
            d for d in os.listdir(self.wav_dir)
            if os.path.isdir(os.path.join(self.wav_dir, d))
        )
        rng = random.Random(params["seed"])
        rng.shuffle(speakers)

        n_train, n_eval = params["n_speakers"], params["n_eval_speakers"]
        train_speakers = speakers[:n_train]
        eval_speakers = speakers[n_train:n_train + n_eval]
        assert set(train_speakers).isdisjoint(eval_speakers), "Speaker split must be disjoint"

        def pick_items(speaker_list, per_speaker):
            items = []
            for spk in speaker_list:
                spk_wav_dir = os.path.join(self.wav_dir, spk)
                wav_files = sorted(f for f in os.listdir(spk_wav_dir) if f.endswith(".wav"))[:per_speaker]
                for wav_file in wav_files:
                    utt_id = wav_file[:-4]  # strip ".wav"
                    items.append({"speaker_id": spk, "utterance_id": utt_id})
            return items

        train_items = pick_items(train_speakers, params["utterances_per_speaker"])
        eval_items = pick_items(eval_speakers, params["eval_utterances_per_speaker"])

        index = {"params": params, "train_items": train_items, "eval_items": eval_items}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"[VCTKSubset] Saved subset index to {index_path} "
              f"({len(train_speakers)} train speakers, {len(eval_speakers)} eval speakers, confirmed disjoint)")
        return index

    def __len__(self):
        return len(self._items)

    def _load_transcript(self, speaker_id: str, utterance_id: str) -> str:
        txt_path = os.path.join(self.txt_dir, speaker_id, f"{utterance_id}.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                return f.read().strip()
        return ""  # some utterances may lack a transcript file -- don't crash, just return empty

    def __getitem__(self, idx):
        item = self._items[idx]
        speaker_id, utterance_id = item["speaker_id"], item["utterance_id"]
        wav_path = os.path.join(self.wav_dir, speaker_id, f"{utterance_id}.wav")

        waveform_np, native_sr = sf.read(wav_path)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)  # collapse to mono if stereo

        if native_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, native_sr, self.sample_rate)

        if waveform.shape[0] >= self.crop_samples:
            if self.split == "train":
                start = random.randint(0, waveform.shape[0] - self.crop_samples)
            else:
                start = (waveform.shape[0] - self.crop_samples) // 2
            waveform = waveform[start:start + self.crop_samples]
        else:
            pad = self.crop_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        transcript = self._load_transcript(speaker_id, utterance_id)
        return {"waveform": waveform.unsqueeze(0), "speaker_id": speaker_id, "transcript": transcript}


def collate_vctk(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])
    speaker_ids = [item["speaker_id"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    return {"waveform": waveforms, "speaker_id": speaker_ids, "transcript": transcripts}
