"""
Lightweight data preparation for autoresearch experiments (CPU edition).
Downloads TinyStories dataset and uses byte-level tokenization (no BPE training needed).

Usage:
    python prepare.py                  # full prep (download dataset)
    python prepare.py --num-shards 2   # download only 2 shards (for quick testing)

Data is stored in ~/.cache/autoresearch_lite/.
"""

import os
import sys
import time
import math
import argparse
from multiprocessing import Pool

import requests
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 256        # context length (small for CPU)
TIME_BUDGET = 120        # training time budget in seconds (2 minutes)
EVAL_TOKENS = 2 * 131072 # number of tokens for val eval (~262K)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_lite")
DATA_DIR = os.path.join(CACHE_DIR, "data")
# TinyStories dataset from HuggingFace (GPT-4 generated short stories, low entropy)
BASE_URL = "https://huggingface.co/api/datasets/roneneldan/TinyStories/parquet/default"
TRAIN_FILES = [
    ("train_0.parquet", "train/0.parquet"),
    ("train_1.parquet", "train/1.parquet"),
    ("train_2.parquet", "train/2.parquet"),
    ("train_3.parquet", "train/3.parquet"),
]
VAL_FILES = [
    ("validation_0.parquet", "validation/0.parquet"),
]

# Byte-level tokenizer: vocab_size = 256 (raw bytes) + 1 BOS token
VOCAB_SIZE = 257
BOS_TOKEN_ID = 256  # special BOS token

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_file(args):
    """Download one file with retries. Returns True on success."""
    filename, url = args
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards=2):
    """Download training shards + validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Select subset of training files: each entry is (local_name, remote_path)
    train_subset = TRAIN_FILES[:min(num_shards, len(TRAIN_FILES))]
    all_files = [(local, f"{BASE_URL}/{remote}") for local, remote in train_subset]
    all_files += [(local, f"{BASE_URL}/{remote}") for local, remote in VAL_FILES]

    existing = sum(1 for f, _ in all_files if os.path.exists(os.path.join(DATA_DIR, f)))
    if existing == len(all_files):
        print(f"Data: all {len(all_files)} files already downloaded at {DATA_DIR}")
        return True

    needed = len(all_files) - existing
    print(f"Data: downloading {needed} files ({existing} already exist)...")

    # Try downloading; if parquet fails, fall back to text
    success_count = 0
    for args in all_files:
        if download_single_file(args):
            success_count += 1

    if success_count < len(all_files):
        print(f"Warning: only {success_count}/{len(all_files)} files downloaded.")
        print("Falling back to generating synthetic training data...")
        generate_fallback_data()
        return True

    print(f"Data: {success_count}/{len(all_files)} files ready at {DATA_DIR}")
    return True


def generate_fallback_data():
    """Generate simple synthetic text data if download fails."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Simple repetitive stories for testing the pipeline
    stories = [
        "Once upon a time, there was a little cat. The cat liked to play in the garden. One day, the cat found a ball. The cat was very happy.",
        "Tom had a red toy car. He liked to drive it around the house. His mom said be careful. Tom smiled and played all day.",
        "Lily went to the park with her dog. The dog ran very fast. Lily laughed and ran too. They had a great time together.",
        "The sun was shining bright. Birds were singing in the trees. A little boy named Sam went outside to play. He loved sunny days.",
        "Anna had a new book. She sat under a big tree and started reading. The story was about a princess. Anna loved reading stories.",
    ]

    # Write train data
    train_path = os.path.join(DATA_DIR, "train.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        for _ in range(2000):  # repeat to get enough data
            for story in stories:
                f.write(story + "\n\n")

    # Write val data
    val_path = os.path.join(DATA_DIR, "val.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        for _ in range(200):
            for story in stories:
                f.write(story + "\n\n")

    print(f"Fallback data generated at {DATA_DIR}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Byte-level tokenizer. No training needed — every byte is a token."""

    def __init__(self):
        self.bos_token_id = BOS_TOKEN_ID

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        return cls()

    def get_vocab_size(self):
        return VOCAB_SIZE

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if isinstance(text, str):
            ids = list(text.encode("utf-8"))
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.bos_token_id
                ids.insert(0, prepend_id)
            return ids
        elif isinstance(text, list):
            result = []
            for t in text:
                ids = list(t.encode("utf-8"))
                if prepend is not None:
                    prepend_id = prepend if isinstance(prepend, int) else self.bos_token_id
                    ids.insert(0, prepend_id)
                result.append(ids)
            return result
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        byte_ids = [i for i in ids if i < 256]
        return bytes(byte_ids).decode("utf-8", errors="replace")


def _load_texts(split):
    """Load text documents from downloaded data."""
    # First try parquet files
    try:
        import pyarrow.parquet as pq
        parquet_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
        if parquet_files:
            if split == "train":
                files = [f for f in parquet_files if "train" in f]
            else:
                files = [f for f in parquet_files if "validation" in f or "val" in f]

            texts = []
            for fname in files:
                filepath = os.path.join(DATA_DIR, fname)
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts.extend(rg.column("text").to_pylist())
            if texts:
                return texts
    except (ImportError, Exception):
        pass

    # Fall back to text files
    txt_file = os.path.join(DATA_DIR, f"{split if split != 'val' else 'val'}.txt")
    if not os.path.exists(txt_file):
        txt_file = os.path.join(DATA_DIR, "train.txt" if split == "train" else "val.txt")
    if os.path.exists(txt_file):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Split on double newlines to get documents
        docs = [d.strip() for d in content.split("\n\n") if d.strip()]
        return docs

    raise FileNotFoundError(f"No data found for split '{split}' in {DATA_DIR}. Run prepare.py first.")


def make_dataloader(tokenizer, B, T, split, buffer_size=500):
    """
    Simple dataloader: packs documents into fixed-length sequences.
    Every row starts with BOS. CPU-only.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()

    texts = _load_texts(split)
    doc_idx = 0

    def get_next_doc():
        nonlocal doc_idx
        doc = texts[doc_idx % len(texts)]
        doc_idx += 1
        epoch = (doc_idx - 1) // len(texts) + 1
        return tokenizer.encode(doc, prepend=bos_token), epoch

    while True:
        rows = []
        epoch = 1
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                tokens, epoch = get_next_doc()
                remaining = row_capacity - len(row)
                row.extend(tokens[:remaining])
            rows.append(row[:row_capacity])

        buf = torch.tensor(rows, dtype=torch.long)
        inputs = buf[:, :-1]
        targets = buf[:, 1:]
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    For byte-level tokenizer: each token IS a byte, so BPB = cross-entropy / ln(2).
    Special tokens (BOS, id >= 256) are excluded.
    """
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        # Mask out special tokens (BOS = 256)
        mask = y_flat < 256
        total_nats += (loss_flat * mask.float()).sum().item()
        total_bytes += mask.sum().item()
    return total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch (CPU lightweight)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of training shards to download (default: 1)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Byte-level tokenizer: vocab_size={VOCAB_SIZE}")
    print()

    download_data(num_shards=args.num_shards)
    print()
    print("Done! Ready to train.")
