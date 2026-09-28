# ColorSAR — Synthetic Aperture Radar (SAR) Image Colorization

An end-to-end deep learning framework for colorizing grayscale satellite imagery (Sentinel-1 SAR / Sentinel-2 optical pairs) using an **Ensemble Encoder-Decoder** architecture in the CIE $L^*a^*b^*$ color space.

---

## 1. Project Overview & Architecture

ColorSAR converts single-channel luminance ($L^*$, brightness) into realistic two-channel chrominance ($a^*, b^*$, color), reconstructing natural RGB satellite imagery.

```
                      +-------------------+
                      |   L Channel (1)   |
                      |   (repeat to 3ch) |
                      +---------+---------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
     +---------------+                     +---------------+
     |   ResNet-50   |                     | DenseNet-121  |
     | (Pretrained)  |                     | (Pretrained)  |
     +-------+-------+                     +-------+-------+
             |                                     |
             +------------------+------------------+
                                |
                                v
                     +--------------------+
                     |  1x1 Convolutions  |
                     |  & Fusion Blocks   |
                     +---------+----------+
                               |
                   Skip Connections (56, 28, 14, 7)
                               |
                               v
                     +--------------------+
                     |    U-Net Style     |
                     |      Decoder       |
                     +---------+----------+
                               |
                               v
                     Predicted (a, b) Channels
                               |
                 Combine with L -> Reconstructed RGB
```

- **Ensemble Encoder**: Combines feature extractors from **ResNet-50** and **DenseNet-121** (ImageNet-pretrained, frozen during Model 1 training). Intermediate multi-scale features are fused using $1 \times 1$ convolutions and batch-normalized fusion blocks.
- **U-Net Decoder**: Transposed convolutions and bilinear upsampling blocks with skip connections restore spatial resolution from $7 \times 7$ back to $224 \times 224 \times 2$.
- **Total Model Parameters**: `45,597,958` (~45.6M).

---

## 2. Model 1 Training & Results (`model1_run/`)

Model 1 (clean-input baseline colorization) was trained across the **full dataset** (~16,000 paired satellite images across all 4 terrain classes: Agricultural, Urban, Barren, and Grassland).

### Training Configuration
- **Hardware**: Kaggle 2x NVIDIA Tesla T4 GPUs (31.2 GB total VRAM) with `torch.nn.DataParallel`
- **Dataset**: `requiemonk/sentinel12-image-pairs-segregated-by-terrain` (v_2)
- **Batch Size**: 64 (32 per GPU)
- **Epochs**: 20
- **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$ with `CosineAnnealingLR` scheduler)
- **Loss Function**: Mean Squared Error (MSE) on normalized $a^*, b^*$ channels

### Convergence & Loss Progression

| Epoch | Train Loss (MSE) | Validation Loss (MSE) | Notes |
| :---: | :---: | :---: | :--- |
| **1** | `0.0384` | `0.0129` | Initial color distribution convergence |
| **5** | `0.0087` | `0.0079` | Rapid reduction in loss |
| **10** | `0.0108` | `0.0075` | Generalization across diverse terrain sets |
| **15** | `0.0060` | `0.0051` | Stable fine-tuning of chromatic details |
| **17** | `0.0056` | **`0.0043`** | **Best Validation Loss reached** |
| **20** | **`0.0049`** | `0.0043` | Final converged checkpoint saved |

### Artifacts in `model1_run/`
- **[`model_1.pth`](model1_run/model_1.pth)**: Trained PyTorch decoder weights (31.3 MB).
- **[`loss_curve.png`](model1_run/loss_curve.png)**: 20-epoch training vs. validation loss curve.
- **[`sample_predictions.png`](model1_run/sample_predictions.png)**: 3-panel visual evaluation (Grayscale Input vs. Ground Truth RGB vs. ColorSAR Prediction).
- **[`history.json`](model1_run/history.json)**: Raw JSON loss logs for each epoch.
- **[`colorized_sample_test.png`](model1_run/colorized_sample_test.png)**: Local inference verification output on `sample_input.jpeg`.

---

## 3. Local Environment & Hardware Verification

The project is configured and verified to run locally on your laptop:

- **Laptop Hardware**:
  - **CPU**: Intel Core i5 (13th Gen)
  - **RAM**: 24 GB DDR5
  - **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, CUDA 12.6 / Driver 581.86)
- **Virtual Environment**: `.venv`
- **Key Installed Packages**:
  - `torch==2.14.0+cu126` & `torchvision==0.29.0+cu126` (GPU enabled)
  - `opencv-python==5.0.0.93`
  - `scikit-image==0.26.0`
  - `matplotlib==3.11.2`
  - `pillow==12.3.0`
  - `numpy==2.5.3`

---

## 4. How to Run Local Inference / Colorization

Use [`colorize_image.py`](colorize_image.py) to test the trained model on any image:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Colorize a sample image using your trained model
python colorize_image.py --image sample_input.jpeg --checkpoint model1_run/model_1.pth --out colorized_result.png
```

This will run inference on your RTX 4050 GPU and generate a 3-way side-by-side comparison:
1. Grayscale $L$ Input
2. ColorSAR Predicted Color
3. Ground Truth Original Image

---

## 5. Repository File Structure

```
ColorSAR/
├── .venv/                         # Virtual environment (PyTorch CUDA 12.6, dependencies)
├── model1_run/                    # Trained Model 1 Checkpoint & Results
│   ├── model_1.pth                # 31.3 MB trained decoder weights
│   ├── loss_curve.png             # Training & validation convergence plot
│   ├── sample_predictions.png     # Validation sample comparisons
│   ├── history.json               # Epoch-by-epoch loss records
│   └── colorized_sample_test.png  # Local verification test on sample_input.jpeg
├── ColorSAR_Kaggle_Training.ipynb # Multi-GPU training notebook for Kaggle (2x T4)
├── colorize_image.py              # Standalone inference & colorization CLI tool
├── model_architecture.py          # EnsembleEncoder, Decoder & ColorizationModel
├── data_pipeline.py               # Lab space conversion & patch extraction logic
├── Train_model1.py                # Local training script (supports subset via --n_images)
├── visual_test.py                 # Pipeline sanity check script
├── requirements.txt               # Dependencies file
├── sample_input.jpeg              # Sample test satellite image
└── README.md                      # Complete project documentation
```

---

## 6. Next Steps

- **Model 2 (SAR Speckle Denoising + Colorization)**: Integrate a **DnCNN** (Denoising Convolutional Neural Network) preprocessing module before the Ensemble Encoder to handle real Sentinel-1 radar speckle noise prior to chromatic prediction.
