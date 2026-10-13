"""Gradio demo: photo -> material -> bin.

  CKPT=runs/effb0/best.pt python app.py
"""
import os

import gradio as gr
import torch

from common import BIN, CLASSES, load

model, tf = load(os.environ.get("CKPT", "best.pt"))
THRESHOLD = 0.6  # below this the app admits it is unsure instead of guessing


@torch.no_grad()
def classify(image):
    if image is None:
        return {}, ""
    prob = model(tf(image.convert("RGB")).unsqueeze(0)).softmax(1)[0]
    top = CLASSES[prob.argmax()]
    if prob.max() < THRESHOLD:
        verdict = "## Not sure, check manually\nTry a closer photo of just the item."
    else:
        verdict = f"## {BIN[top]}\n**{top}**, {prob.max():.0%} confident"
    return {c: float(p) for c, p in zip(CLASSES, prob)}, verdict


demo = gr.Interface(
    classify,
    gr.Image(type="pil", sources=["upload", "webcam"], label="Photo of the item"),
    [gr.Label(num_top_classes=3, label="Material"), gr.Markdown()],
    title="Waste Classifier",
    description="Snap one item. The model predicts its material, then maps it to a bin.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
