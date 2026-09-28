# Model 1 Checkpoint & Evaluation Results

This folder contains the trained weights, loss history, and visual benchmark evaluations for **ColorSAR Model 1**.

---

## 1. Checkpoint Details

- **File**: `model_1.pth`
- **Size**: 31.3 MB (`31,296,058 bytes`)
- **Trained Module**: U-Net Decoder (Ensemble Encoder with ResNet-50 + DenseNet-121 was frozen during training)
- **Architecture**: 5-stage Transposed Convolution & Bilinear Upsampling Decoder with skip connections
- **Input Channels**: Multi-scale fused features ($7\times7, 14\times14, 28\times28, 56\times56$)
- **Output Channels**: 2 channels ($a^*, b^*$ chrominance in normalized range $[-1, 1]$)

---

## 2. Training Run Details

- **Dataset**: `requiemonk/sentinel12-image-pairs-segregated-by-terrain` (Full ~16,000 paired satellite images across `agri`, `urban`, `barren`, `grassland`)
- **Hardware**: 2x NVIDIA Tesla T4 GPUs (31.2 GB VRAM) on Kaggle
- **Parallelization**: `torch.nn.DataParallel` (batch size 64 total, 32 per GPU)
- **Epochs**: 20
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$ with `CosineAnnealingLR` decay down to $1 \times 10^{-5}$)

---

## 3. Loss History Summary

| Epoch | Train Loss | Validation Loss |
| :---: | :---: | :---: |
| 1 | 0.0384 | 0.0129 |
| 2 | 0.0134 | 0.0147 |
| 3 | 0.0112 | 0.0110 |
| 4 | 0.0093 | 0.0099 |
| 5 | 0.0087 | 0.0079 |
| 6 | 0.0089 | 0.0076 |
| 7 | 0.0167 | 0.0171 |
| 8 | 0.0108 | 0.0075 |
| 9 | 0.0084 | 0.0064 |
| 10 | 0.0076 | 0.0061 |
| 11 | 0.0074 | 0.0081 |
| 12 | 0.0063 | 0.0051 |
| 13 | 0.0060 | 0.0052 |
| 14 | 0.0058 | 0.0062 |
| 15 | 0.0056 | 0.0102 |
| 16 | 0.0056 | 0.0051 |
| 17 | 0.0053 | **0.0043** *(Best Checkpoint Saved)* |
| 18 | 0.0051 | 0.0046 |
| 19 | 0.0050 | 0.0047 |
| 20 | **0.0049** | 0.0043 |

---

## 4. Visual Evaluations

1. **`loss_curve.png`**: Displays the 20-epoch loss progression confirming steady convergence without divergence.
2. **`sample_predictions.png`**: Multi-panel validation images showing ground truth optical images vs. ColorSAR predicted colors.
3. **`colorized_sample_test.png`**: Inference test executed locally on the laptop's RTX 4050 GPU using `sample_input.jpeg`.

---

## 5. How to Load in Python

```python
import torch
from model_architecture import EnsembleEncoder, Decoder, ColorizationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = EnsembleEncoder(pretrained=True).to(device)
encoder.eval()

decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("model1_run/model_1.pth", map_location=device))
decoder.eval()

model = ColorizationModel(encoder, decoder).to(device)
model.eval()
```
