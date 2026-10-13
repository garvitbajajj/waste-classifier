# Waste Classifier

Photo of a waste item → **material** (11 classes) → **bin** (green / blue / red / black).
The model is trained on materials, not bins, because materials stay the same everywhere and bin rules change from city to city. To change the bin scheme, edit `BIN` in `common.py`.

## Results

_Fill in from `evaluate.py` once you have your own numbers._

| Model | Trained on | test macro-F1 | **real_test macro-F1** | real_test bin acc | hazardous recall | size |
|---|---|---|---|---|---|---|
| ResNet-18 from scratch | public | | | | | |
| MobileNetV3 | public | | | | | |
| EfficientNet-B0 | public | | | | | |
| EfficientNet-B0 | public + own photos | | | | | |
| Zero-shot vision LLM | — | | | | | — |

`test` is a held-out split of the public data. `real_test` is photos I took myself and never trained on. The gap between those two columns is the domain-shift result.

## Pipeline

```
pip install -r requirements.txt

# 1. Data: put downloaded datasets under raw/<name>/ as they unzip, and your own photos under raw/own/<material>/
python prepare_data.py            # dedupes, remaps labels, splits -> data/
python test_prepare.py            # self-check for the merge/split logic

# 2. Train (on a free Kaggle/Colab GPU)
python train.py --name scratch   --arch resnet18 --no-pretrained
python train.py --name mnv3      --arch mobilenetv3_large_100
python train.py --name effb0     --arch efficientnet_b0
python train.py --name effb0_own --arch efficientnet_b0 --extra data/own_train

# 3. Evaluate: report, confusion matrix, mistakes CSV
python evaluate.py runs/effb0/best.pt data/test data/real_test

# 4. Demo
CKPT=runs/effb0_own/best.pt python app.py
```

## Datasets

Folder names from these datasets are mapped to materials in `REMAP` in `prepare_data.py`. It prints any folder it skipped.

- [TrashNet](https://github.com/garythung/trashnet): 6 classes, white background
- [Garbage Classification, 12 classes](https://www.kaggle.com/datasets/mostafaabla/garbage-classification): includes battery, biological and clothes
- TrashBox: includes e-waste and medical
- RealWaste (UCI): real landfill photos

Check each dataset's licence before any commercial use.

## Classes → bins

| Bin | Materials |
|---|---|
| Green (wet) | organic |
| Blue (dry) | paper, cardboard, plastic, metal, glass, textile |
| Red (domestic hazardous) | battery, e_waste, medical |
| Black (reject) | trash |
