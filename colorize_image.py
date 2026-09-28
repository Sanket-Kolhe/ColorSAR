"""
ColorSAR - Inference / Colorization Script
Uses your trained model_1.pth to colorize any input image.

Usage:
    python colorize_image.py --image sample_input.jpeg --checkpoint model1_run/model_1.pth
"""
import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_architecture import EnsembleEncoder, Decoder, ColorizationModel

def rgb_to_lab(img):
    lab = rgb2lab(img).astype("float32")
    lab = np.transpose(lab, (2, 0, 1))
    L = 2 * lab[[0], ...] / 100 - 1
    ab = 2 * (lab[[1, 2], ...] + 128) / 255 - 1
    return L, ab

def colorize(image_path, checkpoint_path, output_path="colorized_result.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on: {device}")

    assert os.path.exists(checkpoint_path), f"Checkpoint not found at: {checkpoint_path}"
    assert os.path.exists(image_path), f"Input image not found at: {image_path}"

    # Load & preprocess
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size
    img_resized = np.array(original_img.resize((224, 224)))
    
    L_channel, _ = rgb_to_lab(img_resized)
    L_tensor = torch.from_numpy(L_channel).unsqueeze(0).to(device)  # (1, 1, 224, 224)
    L_3ch = L_tensor.repeat(1, 3, 1, 1)                             # (1, 3, 224, 224)

    # Build model & load weights
    print(f"[*] Loading trained weights from: {checkpoint_path}")
    encoder = EnsembleEncoder(pretrained=True).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder = Decoder().to(device)
    weights = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(weights)
    decoder.eval()

    model = ColorizationModel(encoder, decoder).to(device)
    model.eval()

    print("[*] Colorizing image...")
    with torch.no_grad():
        pred_ab = model(L_3ch).cpu().numpy()[0]  # (2, 224, 224)

    # Unscale L and pred_ab back to Lab color ranges
    L_unscaled = (L_channel + 1) * 0.5 * 100
    ab_unscaled = (pred_ab + 1) * 0.5 * 255 - 128

    lab_pred = np.concatenate([L_unscaled, ab_unscaled], axis=0)
    lab_pred = np.transpose(lab_pred, (1, 2, 0))
    rgb_pred = lab2rgb(lab_pred)
    
    # Resize back to original dimensions for final output
    rgb_pred_full = cv2.resize(rgb_pred, (orig_w, orig_h))

    # Side-by-side plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(np.array(original_img.convert("L")), cmap="gray")
    axes[0].set_title("Input (Grayscale L Channel)")
    axes[0].axis("off")

    axes[1].imshow(rgb_pred_full)
    axes[1].set_title("ColorSAR Predicted Color")
    axes[1].axis("off")

    axes[2].imshow(original_img)
    axes[2].set_title("Ground Truth (Original RGB)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"[SUCCESS] Colorization result saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="sample_input.jpeg", help="Path to input image")
    parser.add_argument("--checkpoint", default="model1_run/model_1.pth", help="Path to trained model_1.pth")
    parser.add_argument("--out", default="colorized_result.png", help="Path to save result")
    args = parser.parse_args()

    colorize(args.image, args.checkpoint, args.out)
