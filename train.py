"""Fine-tune a timm backbone on data/train (+ extra folders), keeping the best val macro-F1.

  python train.py --arch efficientnet_b0 --name effb0
  python train.py --arch efficientnet_b0 --name effb0_own --extra data/own_train
  python train.py --arch resnet18 --no-pretrained --name scratch
"""
import argparse
import csv
from pathlib import Path

import timm
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

from common import CLASSES, transform


class Folder(ImageFolder):
    """ImageFolder with a fixed class order, so every split agrees on label ids
    even when a split is missing some classes."""

    def find_classes(self, directory):
        return CLASSES, {c: i for i, c in enumerate(CLASSES)}


def folder(root, tf):
    return Folder(str(root), tf, allow_empty=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).tolist()
        ys += y.tolist()
    return ys, ps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name; output goes to runs/<name>/")
    ap.add_argument("--arch", default="efficientnet_b0", help="any timm model name")
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (baseline)")
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--extra", nargs="*", default=[], help="more ImageFolder dirs to train on, e.g. data/own_train")
    ap.add_argument("--epochs", default=15, type=int)
    ap.add_argument("--warmup", default=2, type=int, help="epochs training only the head before unfreezing")
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    pretrained = not args.no_pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    model = timm.create_model(args.arch, pretrained=pretrained, num_classes=len(CLASSES)).to(device)

    train = ConcatDataset([folder(d, transform(model, training=True)) for d in [args.data / "train", *args.extra]])
    val = folder(args.data / "val", transform(model))
    train_dl = DataLoader(train, args.batch, shuffle=True, num_workers=args.workers, pin_memory=cuda, drop_last=True)
    val_dl = DataLoader(val, args.batch, num_workers=args.workers)

    # Rare classes (battery, medical) get a bigger say in the loss.
    counts = torch.bincount(torch.tensor([y for d in train.datasets for y in d.targets]), minlength=len(CLASSES))
    weights = counts.sum() / (len(CLASSES) * counts.clamp(min=1))
    loss_fn = nn.CrossEntropyLoss(weight=weights.float().to(device), label_smoothing=0.1)

    # Pretrained: train the new head first, then fine-tune everything with a 10x smaller backbone LR.
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr / 10 if pretrained else args.lr},
                             {"params": head, "lr": args.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler(enabled=cuda)
    warmup = args.warmup if pretrained else 0

    out = Path("runs", args.name)
    out.mkdir(parents=True, exist_ok=True)
    if args.wandb:
        import wandb
        wandb.init(project="waste-classifier", name=args.name, config=vars(args))
    print(f"{len(train)} train / {len(val)} val images on {device}; class counts {counts.tolist()}")

    best = -1.0
    with open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_macro_f1"])
        for epoch in range(args.epochs):
            for p in body:
                p.requires_grad = epoch >= warmup
            model.train()
            total, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
                    loss = loss_fn(model(x), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total, n = total + loss.item() * len(y), n + len(y)
            sched.step()

            ys, ps = predict(model, val_dl, device)
            f1 = f1_score(ys, ps, average="macro", labels=sorted(set(ys)), zero_division=0)
            train_loss = total / max(n, 1)
            log.writerow([epoch, f"{train_loss:.4f}", f"{f1:.4f}"])
            f.flush()
            if args.wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": f1})
            mark = ""
            if f1 > best:
                best, mark = f1, "  <- best, saved"
                torch.save({"arch": args.arch, "state": model.state_dict()}, out / "best.pt")
            print(f"epoch {epoch:>2}  loss {train_loss:.4f}  val macro-F1 {f1:.4f}{mark}")
    print(f"best val macro-F1 {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
