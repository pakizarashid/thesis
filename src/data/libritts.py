"""
src/data/libritts.py

SECOND rewrite -- the first version used torchaudio.datasets.LIBRITTS's
built-in downloader, which triggered a real ~7-8GB download. Same lesson as
vctk.py's rewrite: read directly from a mounted Kaggle input dataset instead.

Confirmed directory structure via direct `find` inspection (not assumed):
    <root>/train-clean-100/<speaker_id>/<chapter_id>/<speaker_id>_<chapter_id>_<...>_<...>.wav
    <root>/train-clean-100/<speaker_id>/<chapter_id>/<speaker_id>_<chapter_id>_<...>_<...>.normalized.txt
    <root>/train-clean-100/<speaker_id>/<chapter_id>/<speaker_id>_<chapter_id>_<...>_<...>.original.txt

This matches LibriTTS's standard official release format exactly.

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


class LibriTTSSubset(Dataset):
    def __init__(self, root: str = "/kaggle/input/datasets/pratt3000/libritts/LibriTTS",
                 n_speakers: int = 10, utterances_per_speaker: int = 10, n_eval_speakers: int = 5,
                 eval_utterances_per_speaker: int = 5, sample_rate: int = 16000,
                 crop_seconds: float = 3.0, split: str = "train", subset: str = "train-clean-100",
                 seed: int = 42, index_cache_dir: str = "./data/libritts_index"):
        self.root = root
        self.data_dir = os.path.join(root, subset)
        self.sample_rate = sample_rate
        self.crop_seconds = crop_seconds
        self.crop_samples = int(crop_seconds * sample_rate)
        self.split = split
        self._native_sample_rate = 24000

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"[LibriTTSSubset] {self.data_dir} not found -- confirm the dataset is attached "
                f"via Kaggle's '+ Add Input' panel and this path matches what 'find' showed."
            )

        os.makedirs(index_cache_dir, exist_ok=True)
        index_path = os.path.join(index_cache_dir, "subset_index.json")
        params = {"n_speakers": n_speakers, "utterances_per_speaker": utterances_per_speaker,
                   "n_eval_speakers": n_eval_speakers, "eval_utterances_per_speaker": eval_utterances_per_speaker,
                   "subset": subset, "seed": seed}

        if os.path.exists(index_path):
            with open(index_path) as f:
                cached = json.load(f)
            if cached.get("params") == params:
                print(f"[LibriTTSSubset] Loaded cached subset index from {index_path}")
                self._index = cached
            else:
                print("[LibriTTSSubset] Cached index params don't match requested -- rebuilding.")
                self._index = self._build_index(params, index_path)
        else:
            self._index = self._build_index(params, index_path)

        key = "train_items" if split == "train" else "eval_items"
        self._items = self._index[key]
        n_spk = len(set(item["speaker_id"] for item in self._items))
        print(f"[LibriTTSSubset] Using {len(self._items)} utterances ({split} split, "
              f"{n_spk} speakers, speaker-disjoint from the other split). "
              f"Reading directly from mounted Kaggle input -- zero working-disk cost.")

    def _build_index(self, params, index_path):
        print("[LibriTTSSubset] Building speaker index (one-time directory scan)...")
        speakers = sorted(
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
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
                spk_dir = os.path.join(self.data_dir, spk)
                collected = 0
                for chapter in sorted(os.listdir(spk_dir)):
                    chapter_dir = os.path.join(spk_dir, chapter)
                    if not os.path.isdir(chapter_dir):
                        continue
                    wav_files = sorted(f for f in os.listdir(chapter_dir) if f.endswith(".wav"))
                    for wav_file in wav_files:
                        if collected >= per_speaker:
                            break
                        utt_id = wav_file[:-4]
                        items.append({"speaker_id": spk, "chapter_id": chapter, "utterance_id": utt_id})
                        collected += 1
                    if collected >= per_speaker:
                        break
            return items

        train_items = pick_items(train_speakers, params["utterances_per_speaker"])
        eval_items = pick_items(eval_speakers, params["eval_utterances_per_speaker"])

        index = {"params": params, "train_items": train_items, "eval_items": eval_items}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"[LibriTTSSubset] Saved subset index to {index_path} "
              f"({len(train_speakers)} train speakers, {len(eval_speakers)} eval speakers, confirmed disjoint)")
        return index

    def __len__(self):
        return len(self._items)

    def _load_transcript(self, speaker_id: str, chapter_id: str, utterance_id: str) -> str:
        txt_path = os.path.join(self.data_dir, speaker_id, chapter_id, f"{utterance_id}.normalized.txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                return f.read().strip()
        return ""

    def __getitem__(self, idx):
        item = self._items[idx]
        speaker_id, chapter_id, utterance_id = item["speaker_id"], item["chapter_id"], item["utterance_id"]
        wav_path = os.path.join(self.data_dir, speaker_id, chapter_id, f"{utterance_id}.wav")

        waveform_np, native_sr = sf.read(wav_path)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)

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

        transcript = self._load_transcript(speaker_id, chapter_id, utterance_id)
        return {"waveform": waveform.unsqueeze(0), "speaker_id": speaker_id, "transcript": transcript}


def collate_libritts(batch):
    waveforms = torch.stack([item["waveform"] for item in batch])
    speaker_ids = [item["speaker_id"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    return {"waveform": waveforms, "speaker_id": speaker_ids, "transcript": transcripts}
