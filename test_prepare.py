"""Run: python test_prepare.py"""
import os
import tempfile
from pathlib import Path

from PIL import Image

import prepare_data


def img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (16, 16), os.urandom(256)).save(path)


with tempfile.TemporaryDirectory() as tmp:
    raw, out = Path(tmp, "raw"), Path(tmp, "data")
    for i in range(5):
        img(raw / "kaggle" / "garbage" / "Brown-Glass" / f"{i}.png")     # nested + renamed class
        img(raw / "realwaste" / "Food Organics" / f"{i}.png")
    (raw / "realwaste" / "Food Organics" / "dup.png").write_bytes(
        (raw / "kaggle" / "garbage" / "Brown-Glass" / "0.png").read_bytes())  # cross-dataset duplicate
    img(raw / "trashbox" / "medical" / "syringe" / "0.png")               # label from a parent folder
    img(raw / "mystery" / "Unknown" / "0.png")                            # unmapped -> skipped
    for i in range(4):
        img(raw / "own" / "plastic" / f"{i}.png")
    extra = Path(tmp, "trashbox")                                          # second --raw root
    for i in range(3):
        img(extra / "TrashBox_train_set" / "e-waste" / f"{i}.png")

    prepare_data.main(["--raw", str(raw), str(extra), "--out", str(out)])

    def count(*parts):
        return len(list(Path(out, *parts).rglob("*.png")))

    assert sum(count(s, "glass") for s in ("train", "val", "test")) == 5
    assert sum(count(s, "organic") for s in ("train", "val", "test")) == 5   # duplicate dropped
    assert sum(count(s, "medical") for s in ("train", "val", "test")) == 1
    assert count("real_test", "plastic") == 2 and count("own_train", "plastic") == 2
    assert sum(count(s, "e_waste") for s in ("train", "val", "test")) == 3
    assert count() == 18                                                    # nothing from "mystery"
    print("ok")
