# Waste Classifier

Photo of a waste item → **material** (11 classes) → **bin** (green / blue / red / black).
The model is trained on materials, not bins, because materials stay the same everywhere and bin rules change from city to city. To change the bin scheme, edit `BIN` in `common.py`.

## Results

15 epochs each on a Kaggle T4, data split 80/10/10 per class.

**Run 1, Kaggle Garbage Classification only** (9 classes; 15,515 images, 15,318 after removing 197 duplicates; test set of 1,534):

| Model | test macro-F1 | test bin acc | test hazardous recall |
|---|---|---|---|
| ResNet-18 from scratch | 0.651 | 0.885 | 0.830 |
| MobileNetV3-Large | 0.949 | 0.985 | 0.968 |
| **EfficientNet-B0** | **0.962** | **0.989** | **0.989** |

**Run 2, Kaggle + TrashBox** (all 11 classes including e-waste and medical; 29,797 images, 28,252 after removing 1,545 duplicates; test set of 2,826):

| Model | test macro-F1 | test bin acc | test hazardous recall | e_waste F1 | medical F1 | **real_test macro-F1** |
|---|---|---|---|---|---|---|
| **EfficientNet-B0** | **0.942** | **0.980** | **0.967** | 0.948 | 0.930 | |
| EfficientNet-B0 + own photos | | | | | | |
| Zero-shot vision LLM | | | | | | |

The two runs use different test sets, so their scores aren't directly comparable. Run 2's set is harder: more classes, and two image sources photographed differently.

- ImageNet pretraining adds about 31 macro-F1 points with the same data and the same 15 epochs.
- EfficientNet-B0: 4.0M parameters, 16 MB checkpoint, about 34 ms per image on a laptop CPU.
- Main confusions are glass ↔ plastic ↔ metal (transparent and shiny items) and paper ↔ cardboard. These stay within one bin.
- Some mistakes reflect overlapping labels rather than model errors: medical → cardboard (medicine boxes) and plastic → medical (syringes and gloves are plastic).
- Run 2: 16 of 479 hazardous items went to a non-hazardous bin, mostly e-waste predicted as metal. 28 safe items were flagged as hazardous.
- Known label caveat: TrashBox's "paper" includes tissues and Tetra Paks, which Indian bin rules don't treat as recyclable paper.
- Bin accuracy is higher than material accuracy because most mistakes stay within one bin (for example paper vs cardboard).
- `test` is a held-out split of the public data. `real_test` is photos I took myself and never trained on. The gap between those two columns is the domain-shift result.

## Pipeline

```
pip install -r requirements.txt

# 1. Data: put downloaded datasets under raw/<name>/ as they unzip, and your own photos under raw/own/<material>/
python prepare_data.py            # dedupes, remaps labels, splits -> data/  (--raw accepts several folders)
python test_prepare.py            # self-check for the merge/split logic

# 2. Train (on a free Kaggle/Colab GPU)
python train.py --name scratch   --arch resnet18 --no-pretrained
python train.py --name mnv3      --arch mobilenetv3_large_100
python train.py --name effb0     --arch efficientnet_b0
python train.py --name effb0_own --arch efficientnet_b0 --extra data/own_train

# 3. Evaluate: report, confusion matrix, mistakes CSV
python evaluate.py runs/effb0/best.pt data/test data/real_test

# 4. Demo
python app.py runs/effb0_own/best.pt
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
