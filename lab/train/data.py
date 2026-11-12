"""Prepare the training text: download a slice of FineWeb-Edu and turn it into token files.

What it does
  1. Streams documents from the public dataset HuggingFaceFW/fineweb-edu (sample-10BT):
     educational web pages, the kind of text the modded-nanogpt speedruns train on.
  2. Learns its own small tokenizer (8,192 word-pieces, byte-level BPE) from the first
     50 MB of that text and saves it as tokenizer.json. Token 0 marks a document start.
  3. Writes the first ~1M tokens to val.bin (never trained on, used only to score a
     model) and the rest to train.bin, as 16-bit integers.
  4. Writes meta.json with the token counts and the validation set's bytes per token.
     That number converts a loss into "bits per byte", a score that stays comparable
     even if the tokenizer is changed later.

Why a small vocabulary
  The model scores every vocabulary entry at every position. With GPT-2's 50,257
  entries that one step took two thirds of the compute and 12.7 GB at batch 32 on this
  Mac (measured 2026-11-11), which pushed it into swap. 8,192 entries is what
  Karpathy's autoresearch uses for the same reason.

Run:     uv run python -m lab.train.data [--text-mb 300]
Changes: writes data/fineweb_edu/{train.bin,val.bin,tokenizer.json,meta.json} (about
         150 MB for the default). Downloads only the documents it needs. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "fineweb_edu"
VOCAB_SIZE = 8192
BOS = "<|bos|>"
VAL_TOKENS = 1_000_000
TOKENIZER_TRAIN_BYTES = 50 * 2**20
CHUNK_DOCS = 2_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--text-mb", type=int, default=300, help="megabytes of raw text to download")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download the text.
    texts: list[str] = []
    size = 0
    for doc in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
        texts.append(doc["text"])
        size += len(doc["text"].encode("utf-8"))
        if size >= args.text_mb * 2**20:
            break
    print(f"downloaded {len(texts):,} documents, {size / 2**20:.0f} MB", flush=True)

    # 2. Learn the tokenizer from the first 50 MB.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    sample, seen = [], 0
    for t in texts:
        sample.append(t)
        seen += len(t)
        if seen >= TOKENIZER_TRAIN_BYTES:
            break
    tokenizer.train_from_iterator(sample, trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=[BOS], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.save(str(DATA_DIR / "tokenizer.json"))
    bos = tokenizer.token_to_id(BOS)
    print(f"tokenizer trained: {tokenizer.get_vocab_size()} entries", flush=True)

    # 3. Tokenize. Whole documents go to validation until it is full, so no document is split across the two sets.
    val: list[np.ndarray] = []
    val_count = val_bytes = train_count = 0
    with (DATA_DIR / "train.bin").open("wb") as train_file:
        for i in range(0, len(texts), CHUNK_DOCS):
            chunk = texts[i:i + CHUNK_DOCS]
            for text, enc in zip(chunk, tokenizer.encode_batch(chunk)):
                tokens = np.array([bos, *enc.ids], dtype=np.uint16)
                if val_count < VAL_TOKENS:
                    val.append(tokens)
                    val_count += len(tokens)
                    val_bytes += len(text.encode("utf-8"))
                else:
                    train_file.write(tokens.tobytes())
                    train_count += len(tokens)
    np.concatenate(val).tofile(DATA_DIR / "val.bin")

    # 4. Record what was made.
    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu sample-10BT (first documents of the stream)",
        "tokenizer": "byte-level BPE trained on the first 50 MB (tokenizer.json)",
        "vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": train_count,
        "val_tokens": val_count,
        "val_bytes": val_bytes,
        "val_bytes_per_token": val_bytes / val_count,
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))

    # The streaming download leaves a background thread that blocked interpreter shutdown
    # (2026-11-11: the process sat in PyThread_hang_thread for minutes after meta.json was
    # written). Every file is closed by now, so leave without waiting for that thread.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
