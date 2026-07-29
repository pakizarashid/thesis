"""
src/data/librispeech.py

Loads a deterministic subset of LibriSpeech train-clean-100 for Stage 1
adapter fine-tuning (per the project plan: "few hundred utterances, dozens of
speakers"). Uses torchaudio's built-in LIBRISPEECH dataset for download +
indexing (full train-clean-100 is ~6.3GB; simplest reliable path given your
2.3TB free disk, versus hand-picking individual FLAC files from an OpenSLR
mirror, which is fragile and mirror-structure-dependent).

The subset selection is deterministic (fixed seed) and cached to disk as a
JSON index, so re-running training doesn't reshuffle which utterances are used
-- important for reproducibility and for keeping your "controlled" eval subset
consistent across Stage 1 / Stage 2 runs.
"""

import os
import json
import random
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class LibriSpeechSubset(Dataset):
    """
    Args:
        root: directory to download/cache LibriSpeech into
        n_speakers: how many distinct speakers to include
        utterances_per_speaker: how many utterances per speaker
        sample_rate: target sample rate (LibriSpeech is natively 16kHz, so this
            is normally a no-op, but resampling is applied defensively in case
            that ever changes)
        crop_seconds: fixed-length crop applied per utterance (random crop
            during training, center crop during eval) -- needed because
            batching requires equal-length tensors, and full utterances vary
            widely in length
        seed: fixes which speakers/utterances are selected, and the RNG used
            for train-time random cropping's initial offset table (crop itself
            is re-randomized per __getitem__ call at train time; see split)
        split: 'train' uses random crop offset each call (data augmentation via
            varied crop position); 'eval' uses a fixed center crop for
            reproducible evaluation
        index_cache_path: where to save/load the deterministic subset index.
            Defaults to root/subset_index.json.
    """

    def __init__(
        self,
        root: str = "./data/librispeech",
        n_speakers: int = 30,
        utterances_per_speaker: int = 10,
        n_eval_speakers: int = 5,
        eval_utterances_per_speaker: int = 5,
        sample_rate: int = 16000,
        crop_seconds: float = 3.0,
        seed: int = 42,
        split: str = "train",
        index_cache_path: str = None,
    ):
        """
        split='train' and split='eval' draw from DISJOINT speaker pools (not
        just different utterances from the same speakers) -- both pools are
        non-overlapping slices of one deterministic shuffle over all available
        speakers, so there's no speaker leakage between them by construction.
        This matters for Stage 1 validation: ACC on 'eval' actually measures
        generalization to unseen speakers, not memorization of the training set.
        """
        self.root = root
        self.sample_rate = sample_rate
        self.crop_len = int(crop_seconds * sample_rate)
        self.split = split
        self.seed = seed

        os.makedirs(root, exist_ok=True)
        self.index_cache_path = index_cache_path or os.path.join(root, "subset_index.json")

        print(f"[LibriSpeechSubset] Loading/downloading train-clean-100 into {root} "
              f"(this is a ~6.3GB download on first run, cached after)...")
        self._full_dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url="train-clean-100", download=True
        )

        index = self._build_or_load_index(
            n_speakers, utterances_per_speaker, n_eval_speakers, eval_utterances_per_speaker
        )
        self._subset_indices = index[split]
        # LibriSpeech's extraction layout ({root}/LibriSpeech/{url}/...) is a
        # stable, public part of the download convention -- constructed
        # directly here rather than reaching into torchaudio's internal
        # dataset attributes (whose name varies across versions: _path in some,
        # _archive in others, as seen in your traceback).
        self._data_root = os.path.join(self.root, "LibriSpeech", "train-clean-100")
        print(f"[LibriSpeechSubset] Using {len(self._subset_indices)} utterances "
              f"({split} split, speaker-disjoint from the other split).")

    def _build_or_load_index(self, n_speakers: int, utterances_per_speaker: int,
                              n_eval_speakers: int, eval_utterances_per_speaker: int):
        params = dict(
            n_speakers=n_speakers, utterances_per_speaker=utterances_per_speaker,
            n_eval_speakers=n_eval_speakers, eval_utterances_per_speaker=eval_utterances_per_speaker,
            seed=self.seed,
        )
        if os.path.exists(self.index_cache_path):
            with open(self.index_cache_path) as f:
                cached = json.load(f)
            if cached.get("params") == params:
                print(f"[LibriSpeechSubset] Loaded cached subset index from {self.index_cache_path}")
                return {"train": cached["train"], "eval": cached["eval"]}
            print(f"[LibriSpeechSubset] Cached index params don't match requested -- rebuilding.")

        print("[LibriSpeechSubset] Building speaker index (one-time scan)...")
        speaker_to_indices = {}
        for idx, fileid in enumerate(self._full_dataset._walker):
            speaker_id = fileid.split("-")[0]
            speaker_to_indices.setdefault(speaker_id, []).append(idx)

        rng = random.Random(self.seed)
        all_speakers = sorted(speaker_to_indices.keys())
        rng.shuffle(all_speakers)

        total_needed = n_speakers + n_eval_speakers
        if total_needed > len(all_speakers):
            raise ValueError(
                f"Requested {n_speakers} train + {n_eval_speakers} eval speakers "
                f"= {total_needed}, but only {len(all_speakers)} speakers exist "
                f"in train-clean-100."
            )
        train_speakers = all_speakers[:n_speakers]
        eval_speakers = all_speakers[n_speakers:n_speakers + n_eval_speakers]
        assert set(train_speakers).isdisjoint(eval_speakers), \
            "BUG: train/eval speaker pools overlap -- this should be impossible given disjoint slicing"

        def build_indices(speakers, per_speaker):
            out = []
            for spk in speakers:
                spk_indices = speaker_to_indices[spk]
                rng.shuffle(spk_indices)
                out.extend(spk_indices[:per_speaker])
            return out

        train_indices = build_indices(train_speakers, utterances_per_speaker)
        eval_indices = build_indices(eval_speakers, eval_utterances_per_speaker)

        with open(self.index_cache_path, "w") as f:
            json.dump({"params": params, "train": train_indices, "eval": eval_indices}, f)
        print(f"[LibriSpeechSubset] Saved subset index to {self.index_cache_path} "
              f"({len(train_speakers)} train speakers, {len(eval_speakers)} eval speakers, "
              f"confirmed disjoint)")
        return {"train": train_indices, "eval": eval_indices}

    def __len__(self):
        return len(self._subset_indices)

    def __getitem__(self, i):
        full_idx = self._subset_indices[i]
        fileid = self._full_dataset._walker[full_idx]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        # Bypass torchaudio.datasets.LIBRISPEECH.__getitem__ (and therefore
        # torchaudio.load(), which defaults to a torchcodec backend requiring
        # working FFmpeg+CUDA linkage -- broken in some environments, see
        # librispeech.py's module-level notes). LibriSpeech's on-disk layout is
        # a long-stable public convention, so we build the path directly and
        # load with soundfile instead, which has no such dependency.
        flac_path = os.path.join(
            self._data_root, speaker_id, chapter_id,
            f"{speaker_id}-{chapter_id}-{utterance_id}.flac"
        )
        wav_np, sr = sf.read(flac_path, dtype="float32")
        waveform = torch.from_numpy(wav_np)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T] mono
        else:
            waveform = waveform.T[:1]  # first channel if unexpectedly multi-channel

        transcript = self._get_transcript(speaker_id, chapter_id, utterance_id)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        waveform = self._crop_or_pad(waveform.squeeze(0))  # [crop_len]

        return {
            "waveform": waveform.unsqueeze(0),  # [1, crop_len]
            "speaker_id": speaker_id,
            "utterance_id": fileid,
            "transcript": transcript,
        }

    def _get_transcript(self, speaker_id: str, chapter_id: str, utterance_id: str) -> str:
        """Reads the per-chapter .trans.txt file (LibriSpeech convention:
        '{speaker}-{chapter}.trans.txt', one line per utterance:
        '{speaker}-{chapter}-{utterance} TRANSCRIPT TEXT'). Cached per-chapter
        to avoid re-reading the same file for every utterance in it."""
        cache_key = (speaker_id, chapter_id)
        if not hasattr(self, "_transcript_cache"):
            self._transcript_cache = {}
        if cache_key not in self._transcript_cache:
            trans_path = os.path.join(
                self._data_root, speaker_id, chapter_id,
                f"{speaker_id}-{chapter_id}.trans.txt"
            )
            chapter_transcripts = {}
            if os.path.exists(trans_path):
                with open(trans_path) as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            chapter_transcripts[parts[0]] = parts[1]
            self._transcript_cache[cache_key] = chapter_transcripts

        fileid = f"{speaker_id}-{chapter_id}-{utterance_id}"
        return self._transcript_cache[cache_key].get(fileid, "")

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        T = waveform.shape[-1]
        if T < self.crop_len:
            pad = self.crop_len - T
            return torch.nn.functional.pad(waveform, (0, pad))
        if T == self.crop_len:
            return waveform
        if self.split == "train":
            start = random.randint(0, T - self.crop_len)
        else:
            start = (T - self.crop_len) // 2  # center crop for eval determinism
        return waveform[start:start + self.crop_len]


def collate_librispeech(batch):
    """Simple collate: waveforms are already fixed-length from _crop_or_pad,
    so this is just a stack + metadata passthrough."""
    waveforms = torch.stack([item["waveform"] for item in batch])  # [B, 1, T]
    return {
        "waveform": waveforms,
        "speaker_id": [item["speaker_id"] for item in batch],
        "utterance_id": [item["utterance_id"] for item in batch],
        "transcript": [item["transcript"] for item in batch],
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_ds = LibriSpeechSubset(n_speakers=5, utterances_per_speaker=3,
                                  n_eval_speakers=2, eval_utterances_per_speaker=2, split="train")
    eval_ds = LibriSpeechSubset(n_speakers=5, utterances_per_speaker=3,
                                 n_eval_speakers=2, eval_utterances_per_speaker=2, split="eval")
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    train_speakers = set(item["speaker_id"] for item in [train_ds[i] for i in range(len(train_ds))])
    eval_speakers = set(item["speaker_id"] for item in [eval_ds[i] for i in range(len(eval_ds))])
    print(f"Train speakers: {train_speakers}")
    print(f"Eval speakers: {eval_speakers}")
    print(f"Disjoint: {train_speakers.isdisjoint(eval_speakers)}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_librispeech)
    batch = next(iter(loader))
    print(f"Batch waveform shape: {batch['waveform'].shape}")
