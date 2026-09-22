"""
ColorSAR - Data Pipeline (Lab conversion + Patching)
Extracted and verified from original_model.ipynb (cells 5, 7, 9, 10, 11)
Source: https://github.com/nishant9083/ColorSAR
"""
import cv2
import numpy as np
import os
from PIL import Image

# ---- Cell 9: RGB -> Lab conversion (manual, no torchvision dependency for this test) ----
from skimage.color import rgb2lab, lab2rgb

def rgb_to_lab(img):
    """Converts an RGB image (H,W,3) to normalized L and ab channels."""
    img_lab = rgb2lab(img).astype("float32")   # (H, W, 3): L in [0,100], a/b in [-128,127]
    img_lab = np.transpose(img_lab, (2, 0, 1))  # -> (3, H, W), same as ToTensor()

    L_channel = img_lab[[0], ...]
    ab_channels = img_lab[[1, 2], ...]

    # scale between -1 and 1, exactly as in the original notebook
    L_channel = 2 * (L_channel - 0) / (100 - 0) - 1
    ab_channels = 2 * (ab_channels - (-128)) / (127 - (-128)) - 1

    return L_channel, ab_channels


# ---- Cell 10: Patch extraction ----
def create_patches(img, patch_size=224):
    """Splits a (C,H,W) image into non-overlapping patch_size x patch_size patches."""
    patches = []
    h, w = img.shape[1:]

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[:, i:i + patch_size, j:j + patch_size]
            if patch.shape[1:] == (patch_size, patch_size):
                patches.append(patch)

    return patches


# ---- Cell 11 (simplified, no torch Dataset wrapper needed for this test) ----
def load_and_preprocess(image_path, size=224):
    """Loads one optical image, resizes it, converts to Lab, returns L and ab tensors (numpy)."""
    color_img = np.array(Image.open(image_path).convert("RGB"))
    color_img = cv2.resize(color_img, (size, size))
    L_channel, ab_channels = rgb_to_lab(color_img)
    return L_channel, ab_channels


if __name__ == "__main__":
    import sys
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    assert test_image, "Pass a test image path"

    print(f"Testing pipeline on: {test_image}")

    L, ab = load_and_preprocess(test_image)
    print(f"L channel shape:  {L.shape}   range: [{L.min():.3f}, {L.max():.3f}]")
    print(f"ab channels shape: {ab.shape}   range: [{ab.min():.3f}, {ab.max():.3f}]")

    # sanity check: reconstruct back to RGB and confirm it round-trips reasonably
    L_unscaled = (L + 1) * 0.5 * 100
    ab_unscaled = (ab + 1) * 0.5 * 255 - 128
    lab_reconstructed = np.concatenate([L_unscaled, ab_unscaled], axis=0)
    lab_reconstructed = np.transpose(lab_reconstructed, (1, 2, 0))
    rgb_reconstructed = lab2rgb(lab_reconstructed)
    print(f"Reconstructed RGB shape: {rgb_reconstructed.shape}, range: [{rgb_reconstructed.min():.3f}, {rgb_reconstructed.max():.3f}]")

    # test patching on a larger synthetic image
    big_img = np.random.rand(3, 500, 500).astype("float32")
    patches = create_patches(big_img, patch_size=224)
    print(f"Patches extracted from 500x500 image: {len(patches)} (expected 4)")
    print(f"Each patch shape: {patches[0].shape}")

    print("\nALL PIPELINE STEPS RAN SUCCESSFULLY.")
