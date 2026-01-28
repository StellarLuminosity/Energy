#!/usr/bin/env python3
"""
Repack teacher logprob cache chunks into a single HuggingFace DatasetDict(train/test).

Typical use:
  python repack_logprob_cache.py \
    --input_dir /scratch/klambert/dataset/logprob_cache_temp2/teacher_logprobs_train \
    --output_dir /scratch/klambert/dataset/logprob_cache_temp2/teacher_logprobs \
    --test_size 0.01 \
    --seed 42 \
    --shuffle

What it does:
  1) Finds chunk_* subdirs
  2) Skips incomplete/corrupt chunks safely
  3) Loads each chunk with datasets.load_from_disk
  4) Concatenates all chunks into one Dataset
  5) Optionally shuffles
  6) Splits into train/test
  7) Saves a DatasetDict to disk
"""

import argparse
import glob
import os
import re
import shutil
from typing import List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets


_CHUNK_RE = re.compile(r"chunk_(\d+)$")


def _sorted_chunk_dirs(input_dir: str) -> List[str]:
    candidates = [
        p for p in glob.glob(os.path.join(input_dir, "chunk_*"))
        if os.path.isdir(p) and _CHUNK_RE.search(os.path.basename(p))
    ]

    def _key(p: str) -> int:
        m = _CHUNK_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else 10**18

    return sorted(candidates, key=_key)


def _looks_complete_chunk(chunk_dir: str) -> bool:
    # Minimum signals of a saved HF dataset directory
    info = os.path.join(chunk_dir, "dataset_info.json")
    arrows = glob.glob(os.path.join(chunk_dir, "data-*.arrow"))
    state = os.path.join(chunk_dir, "state.json")
    return os.path.exists(info) and os.path.exists(state) and len(arrows) > 0


def _safe_load_chunk(chunk_dir: str) -> Optional[Dataset]:
    try:
        ds = load_from_disk(chunk_dir)
    except Exception as e:
        print(f"[skip] failed to load {chunk_dir}: {type(e).__name__}: {e}")
        return None

    if len(ds) == 0:
        print(f"[skip] empty dataset in {chunk_dir}")
        return None

    return ds


def _ensure_same_schema(datasets_list: List[Dataset]) -> List[Dataset]:
    """
    Concatenation requires identical features. If something differs, we try to cast
    to the first dataset's features. If that fails, raise with a clear error.
    """
    if not datasets_list:
        return datasets_list

    target_features = datasets_list[0].features
    fixed = [datasets_list[0]]

    for i, ds in enumerate(datasets_list[1:], start=1):
        if ds.features == target_features:
            fixed.append(ds)
            continue
        try:
            fixed.append(ds.cast(target_features))
            print(f"[warn] chunk #{i}: features differed; casted to match the first chunk schema.")
        except Exception as e:
            raise RuntimeError(
                "Schema mismatch across chunks and automatic casting failed.\n"
                f"- First chunk features: {target_features}\n"
                f"- This chunk features:  {ds.features}\n"
                f"- Chunk index: {i}\n"
                f"- Underlying error: {type(e).__name__}: {e}\n"
                "Fix by ensuring all chunks were produced by the same caching code/config."
            )
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing chunk_* subdirectories (e.g., teacher_logprobs_train).")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Where to save the final DatasetDict (train/test).")
    ap.add_argument("--test_size", type=float, default=0.01,
                    help="Fraction for test split (e.g., 0.01 = 1%).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle before splitting (recommended).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output_dir if it exists.")
    ap.add_argument("--max_chunks", type=int, default=None,
                    help="Optional limit for debugging.")
    args = ap.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir}")

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"output_dir already exists: {args.output_dir}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    chunk_dirs = _sorted_chunk_dirs(input_dir)
    if args.max_chunks is not None:
        chunk_dirs = chunk_dirs[: args.max_chunks]

    print(f"[info] Found {len(chunk_dirs)} chunk dirs under: {input_dir}")

    loaded: List[Dataset] = []
    skipped_incomplete = 0

    for i, cdir in enumerate(chunk_dirs):
        if not _looks_complete_chunk(cdir):
            print(f"[skip] incomplete chunk on disk: {cdir}")
            skipped_incomplete += 1
            continue

        ds = _safe_load_chunk(cdir)
        if ds is None:
            continue

        # Lightweight per-chunk sanity print
        print(f"[load] {os.path.basename(cdir)}: rows={len(ds)} cols={ds.column_names}")
        loaded.append(ds)

    if not loaded:
        raise RuntimeError(
            "No valid chunks were loaded. "
            "Either the directory is wrong, or all chunks are incomplete/corrupt."
        )

    if skipped_incomplete:
        print(f"[warn] Skipped {skipped_incomplete} incomplete chunk dirs.")

    # Ensure schemas match and concatenate
    loaded = _ensure_same_schema(loaded)
    print(f"[info] Concatenating {len(loaded)} chunk datasets ...")
    full: Dataset = concatenate_datasets(loaded)

    print(f"[info] Full dataset rows={len(full)}")
    print(f"[info] Features:\n{full.features}")

    if args.shuffle:
        print(f"[info] Shuffling with seed={args.seed} ...")
        full = full.shuffle(seed=args.seed)

    print(f"[info] Splitting train/test with test_size={args.test_size} seed={args.seed} ...")
    dsdict: DatasetDict = full.train_test_split(test_size=args.test_size, seed=args.seed)

    print(f"[info] Split sizes: train={len(dsdict['train'])} test={len(dsdict['test'])}")
    print(f"[info] Saving DatasetDict to: {args.output_dir}")
    dsdict.save_to_disk(args.output_dir)

    print("[done] Saved.")


if __name__ == "__main__":
    main()
