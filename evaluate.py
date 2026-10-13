"""Evaluate a checkpoint on one or more ImageFolder splits.

  python evaluate.py runs/effb0/best.pt data/test data/real_test

Per split: per-class report, macro-F1, bin accuracy, hazardous recall, plus (next to the
checkpoint) a confusion-matrix PNG and mistakes_<split>.csv, most confident mistakes first.
"""
import csv
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from torch.utils.data import DataLoader

from common import BIN, CLASSES, HAZARDOUS, load
from train import folder

matplotlib.use("Agg")


@torch.no_grad()
def main(ckpt, *splits):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tf = load(ckpt, device)
    out = Path(ckpt).parent
    for split in splits:
        ds = folder(split, tf)
        ys, ps, confs = [], [], []
        for x, y in DataLoader(ds, 64):
            conf, pred = model(x.to(device)).softmax(1).max(1)
            ys += y.tolist()
            ps += pred.tolist()
            confs += conf.tolist()
        present = sorted(set(ys))                # macro-F1 only over classes this split contains
        seen = sorted(set(ys) | set(ps))         # report/matrix also show wrongly predicted classes
        names = [CLASSES[i] for i in seen]
        name = Path(split).name

        hazardous = [CLASSES[p] in HAZARDOUS for y, p in zip(ys, ps) if CLASSES[y] in HAZARDOUS]
        print(f"\n=== {split} ({len(ys)} images) ===")
        print(classification_report(ys, ps, labels=seen, target_names=names, digits=3, zero_division=0))
        print(f"macro-F1          {f1_score(ys, ps, average='macro', labels=present, zero_division=0):.3f}")
        print(f"bin accuracy      {sum(BIN[CLASSES[y]] == BIN[CLASSES[p]] for y, p in zip(ys, ps)) / len(ys):.3f}")
        print(f"hazardous recall  {sum(hazardous) / len(hazardous):.3f}" if hazardous else "hazardous recall  n/a")

        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay.from_predictions(ys, ps, labels=seen, display_labels=names, normalize="true",
                                                values_format=".2f", xticks_rotation=45, colorbar=False, ax=ax)
        ax.set_title(f"{name}: rows = true class, normalised")
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name}.png", dpi=120)
        plt.close(fig)

        mistakes = sorted(((c, ds.samples[i][0], CLASSES[y], CLASSES[p])
                           for i, (y, p, c) in enumerate(zip(ys, ps, confs)) if y != p), reverse=True)
        with open(out / f"mistakes_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["confidence", "path", "true", "pred"])
            w.writerows((f"{c:.3f}", path, t, p) for c, path, t, p in mistakes)
        print(f"wrote confusion_{name}.png and mistakes_{name}.csv ({len(mistakes)} mistakes) to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
