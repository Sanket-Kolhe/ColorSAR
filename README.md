# ColorSAR — Reproduction Verification Kit

Prepared for mid-sem checkpoint. This confirms which parts of the original
project's code (github.com/nishant9083/ColorSAR) we have independently
extracted, run, and verified — separate from the original authors' own
results/repo.

## What's in this folder

- `data_pipeline.py` — standalone, runnable extraction of the dataset
  loading + Lab color space conversion + patching logic (from
  `original_model.ipynb`, cells 5/7/9/10/11).
- `model_architecture.py` — standalone, runnable extraction of the
  EnsembleEncoder (ResNet50 + DenseNet121 fusion) + Decoder + full
  ColorizationModel (from `original_model.ipynb`, cells 15/17/19).
- `data_pipeline_log.txt` — actual console output from running
  `data_pipeline.py` tonight, on a real sample image from the repo.
- `model_architecture_log.txt` — actual console output from running
  `model_architecture.py` tonight.
- `sample_input.jpeg` — the sample image used for the data pipeline test.

## What this verifies (genuinely true, run tonight)

1. **Data/Lab pipeline runs correctly end-to-end**: image load → resize →
   RGB→Lab conversion → channel normalization to [-1, 1] → reconstruction
   back to RGB (sanity round-trip check) → patch extraction into 224x224
   tiles. All steps executed without error on a real image.

2. **Model architecture is structurally correct**: the EnsembleEncoder
   (fusing ResNet50 + DenseNet121 features via 1x1 convs and fusion
   blocks) and the U-Net-style Decoder were instantiated (45.6M
   parameters) and a forward pass was run on a dummy 224x224x3 input,
   producing the expected 224x224x2 (a,b channel) output shape.

   Note: this test used `pretrained=False` (random-initialized weights)
   to keep it fast and lightweight — it verifies the architecture's
   shapes/wiring are correct, not the actual colorization quality. Real
   training requires the ImageNet-pretrained weights (as the original
   notebook does) and the full Kaggle dataset.

## What this does NOT cover yet (honest gaps — the real "remaining work")

- **Full Kaggle dataset was not downloaded in this environment** (network
  restrictions here block kaggle.com). The team needs to run
  `!kaggle datasets download -d requiemonk/sentinel12-image-pairs-segregated-by-terrain`
  in Colab/Kaggle/local machine with valid Kaggle API credentials.
- **No actual training was run** — this only confirms the code runs
  without crashing, not that it produces the original's reported
  PSNR/SSIM numbers. That requires full training on GPU with the real
  dataset (several hours per model, per the original repo).
- **DnCNN denoising module** (used for Model 2 / gray_model.ipynb) was
  not tested in this pass — only the Model 1 (clean-input) path was
  verified. Recommend testing this next.

## Suggested framing for the mid-sem slide

"We have extracted and independently verified the data preprocessing
pipeline and model architecture from the original ColorSAR repository —
confirming both run correctly end-to-end on our side. Full model
training on the complete dataset is the remaining work before final
submission."
