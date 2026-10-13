"""Merge public waste datasets into one material-labelled ImageFolder layout.

  raw/<any dataset>/**/<class folder>/**/*.jpg  ->  data/{train,val,test}/<material>/
  raw/own/<material>/*.jpg                      ->  data/{own_train,real_test}/<material>/

Duplicates (same perceptual hash) are dropped across *all* sources before splitting,
so the same picture can never sit in both train and test.
"""
import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import CLASSES

# Lower-cased folder names from TrashNet, Kaggle garbage-12, TrashBox, RealWaste -> material.
REMAP = {c: c for c in CLASSES} | {
    "biological": "organic", "food organics": "organic", "vegetation": "organic",
    "brown-glass": "glass", "green-glass": "glass", "white-glass": "glass",
    "clothes": "textile", "shoes": "textile", "textile trash": "textile",
    "e-waste": "e_waste", "miscellaneous trash": "trash",
}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(path):
    """64-bit difference hash: survives resizing and re-compression."""
    im = Image.open(path)
    im.draft("L", (64, 64))  # JPEGs decode at a fraction of full size: much faster on big photos
    px = im.convert("L").resize((9, 8)).tobytes()
    return sum(1 << i for i in range(64) if px[i + i // 8] > px[i + i // 8 + 1])


def label_for(parts):
    """Nearest folder (walking up from the file) whose name we know."""
    return next((REMAP[p.lower()] for p in reversed(parts) if p.lower() in REMAP), None)


def collect(raw):
    public, own, seen, unknown = defaultdict(list), defaultdict(list), set(), Counter()
    paths = [p for p in sorted(raw.rglob("*")) if p.suffix.lower() in IMAGE_EXT and p.parent != raw]
    print(f"found {len(paths)} images under {raw}", flush=True)
    for i, p in enumerate(paths, 1):
        if i % 1000 == 0:
            print(f"  hashed {i}/{len(paths)}", flush=True)
        dataset, *folders = p.relative_to(raw).parts[:-1]
        label = label_for(folders)
        if label is None:
            unknown[p.parent.relative_to(raw).as_posix()] += 1
            continue
        try:
            h = dhash(p)
        except OSError:  # corrupt / truncated file
            continue
        # ponytail: exact hash match only; near-duplicates (hamming <= 4) need a BK-tree, add if leakage shows up
        if h in seen:
            continue
        seen.add(h)
        (own if dataset == "own" else public)[label].append((p, h))
    return public, own, unknown


def write(items, out):
    out.mkdir(parents=True, exist_ok=True)
    for p, h in items:
        shutil.copy2(p, out / f"{h:016x}{p.suffix.lower()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="raw", type=Path)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--val", default=0.1, type=float)
    ap.add_argument("--test", default=0.1, type=float)
    ap.add_argument("--own-test", default=0.5, type=float, help="share of your own photos held out as real_test")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args(argv)

    public, own, unknown = collect(args.raw)
    if args.out.exists():
        shutil.rmtree(args.out)  # generated output only; raw/ is never touched
    rng = random.Random(args.seed)

    for label, items in public.items():
        rng.shuffle(items)
        n_val, n_test = round(len(items) * args.val), round(len(items) * args.test)
        write(items[:n_val], args.out / "val" / label)
        write(items[n_val:n_val + n_test], args.out / "test" / label)
        write(items[n_val + n_test:], args.out / "train" / label)
    for label, items in own.items():
        rng.shuffle(items)
        k = round(len(items) * args.own_test)
        write(items[:k], args.out / "real_test" / label)
        write(items[k:], args.out / "own_train" / label)

    print(f"{'class':<10} {'public':>7} {'own':>5}")
    for c in CLASSES:
        print(f"{c:<10} {len(public[c]):>7} {len(own[c]):>5}")
    if unknown:
        print("\nSkipped folders (add them to REMAP if they hold useful classes):")
        for folder, n in unknown.most_common():
            print(f"  {n:>6}  {folder}")


if __name__ == "__main__":
    main()
