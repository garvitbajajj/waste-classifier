"""Run: python test_prepare.py"""
import os
import tempfile
from pathlib import Path

from PIL import Image

import prepare_data


def img(path, size=(16, 16), exif=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    im.save(path, **({"exif": exif} if exif else {}))


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
    sideways = Image.Exif()
    sideways[0x0112] = 6                                                   # phone photo: "rotate 90° to view"
    for i in range(2):                                                     # own photos inside a Kaggle dataset
        img(extra / "my-photos" / "own" / "metal" / f"{i}.jpg", size=(40, 20), exif=sideways)

    prepare_data.main(["--raw", str(raw), str(extra), "--out", str(out)])

    def files(*parts):
        return list(Path(out, *parts).rglob("*.jpg"))

    def count(*parts):
        return len(files(*parts))

    assert sum(count(s, "glass") for s in ("train", "val", "test")) == 5
    assert sum(count(s, "organic") for s in ("train", "val", "test")) == 5   # duplicate dropped
    assert sum(count(s, "medical") for s in ("train", "val", "test")) == 1
    assert sum(count(s, "e_waste") for s in ("train", "val", "test")) == 3
    assert count("real_test", "plastic") == 2 and count("own_train", "plastic") == 2
    assert count("real_test", "metal") == 1 and count("own_train", "metal") == 1  # nested "own" folder
    assert all(Image.open(f).size == (20, 40) for f in files("real_test", "metal"))  # stored upright
    assert count() == 20                                                    # nothing from "mystery"
    print("ok")
