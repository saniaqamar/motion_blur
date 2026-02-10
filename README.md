# Blur Detection System

A deep learning solution that classifies images as sharp or blurred using transfer learning and hybrid features. Achieves 94.29% accuracy by combining CNN features with traditional computer vision metrics.

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python train.py
```

Run inference:
```bash
python predict.py /path/to/images
```

## What This Does

The system detects blur in images for quality control applications. It uses:
- MobileNetV2 as the base model (transfer learning)
- Traditional CV features (Laplacian variance, edges, gradients)
- SVM classifier on combined features

Tested on 966 images with 3 classes: sharp, motion blurred, and defocused blurred.

## Performance

| Model | Accuracy |
|-------|----------|
| CNN Baseline | 92.86% |
| SVM Hybrid | 94.29% |
| Random Forest Hybrid | 93.33% |
| Ensemble | 93.81% |

Best model: SVM Hybrid with 98.57% recall for blurred images (catches almost all blurry photos).

## Dataset Structure

```
motion_blur/
  blur_dataset/
    blur_dataset/
      sharp/
      motion_blurred/
      defocused_blurred/
```

## Inference Options

```bash
# Default (SVM Hybrid - recommended)
python predict.py images_folder

# CNN only (faster, 92.86% accuracy)
python predict.py images_folder --model cnn

# All models
python predict.py images_folder --model svm_hybrid
python predict.py images_folder --model rf_hybrid
python predict.py images_folder --model ensemble

# Custom output
python predict.py images_folder --output results.csv
```

Output is a CSV file with filename, prediction (0=sharp, 1=blurred), confidence, and model used.

## Key Features

- Laplacian variance is the strongest blur indicator (88% difference sharp vs blurred)
- Models handle both motion blur and defocus blur
- Fast inference: 25-30ms per image on CPU
- Lightweight: ~14 MB total model size
- Multiple model options for speed/accuracy tradeoff

## Files

- `train.py`: Training script with hybrid feature extraction
- `predict.py`: Inference script supporting 4 models
- `blur_detection.ipynb`: Complete analysis and experiments
- `blur_model.h5`: Trained CNN (created after training)
- `svm_hybrid_model.pkl`: Trained SVM (created after training)
- `scaler_*.pkl`: Feature normalizers (created after training)

## Training Details

- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Early stopping at epoch 9
- Data augmentation: rotation, shifts, zoom
- Runtime: 20-40 min on GPU, 2-4 hours on CPU

## Troubleshooting

**Models missing:** Run `python train.py` first

**Out of memory:** Edit train.py and change batch_size from 32 to 16

**Dataset not found:** Check directory structure matches the expected layout

## Use Cases

- Vehicle inspection (reject blurry damage photos)
- Quality control (validate product images)
- Document scanning (flag blurry scans)
- Photo uploads (e-commerce, real estate)
- Medical imaging (detect motion artifacts)

## Status

Successfully tested on 966-image dataset with 94.29% accuracy.
