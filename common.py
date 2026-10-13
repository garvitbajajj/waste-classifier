"""Labels and model loading shared by every script.

The model predicts a *material*; the bin is a lookup, so a new city only means editing BIN.
"""

CLASSES = ["battery", "cardboard", "e_waste", "glass", "medical", "metal",
           "organic", "paper", "plastic", "textile", "trash"]

# Indian SWM Rules 2016 style colours. Edit for your city.
BIN = {
    "organic": "Green (wet waste)",
    "paper": "Blue (dry waste)", "cardboard": "Blue (dry waste)", "plastic": "Blue (dry waste)",
    "metal": "Blue (dry waste)", "glass": "Blue (dry waste)", "textile": "Blue (dry waste)",
    "battery": "Red (domestic hazardous)", "e_waste": "Red (domestic hazardous)",
    "medical": "Red (domestic hazardous)",
    "trash": "Black (reject / landfill)",
}
HAZARDOUS = {c for c, b in BIN.items() if b.startswith("Red")}
assert set(BIN) == set(CLASSES)


def transform(model, training=False):
    import timm.data
    return timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=training)


def load(ckpt_path, device="cpu"):
    import timm, torch
    ckpt = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(ckpt["arch"], num_classes=len(CLASSES))
    model.load_state_dict(ckpt["state"])
    return model.eval().to(device), transform(model)
