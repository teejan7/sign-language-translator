# 🤟 ASL Sign Language Recognition — Training Pipeline

A lightweight, real-time American Sign Language (ASL) recognition system.
This repository contains the **model training pipeline** for the ASL Vision project.

🚀 **Live Demo:** [teejan7-asl-vision.hf.space](https://teejan7-asl-vision.hf.space)

---

## 📁 Repository Structure

```
sign-language-translator/
│
├── train.py                 ← Main training entry point
├── config.py                ← All training configuration
├── data_loader.py           ← Loads ASL dataset images
├── feature_extractor.py     ← 91-D geometric feature extraction
├── model_trainer.py         ← Random Forest training + evaluation
├── model_io.py              ← Save/load model and encoder
├── requirements_train.txt   ← Training dependencies
│
├── asl_alphabet_train/      ← Dataset (87,000 images, 29 classes)
└── models/                  ← Saved model output
    ├── rf_model_68.pkl
    └── label_encoder.pkl
```

---

## 🧠 Model Architecture

```
87,000 ASL Images (29 classes)
         ↓
MediaPipe Hand Landmark Detection
(21 3D landmarks per hand)
         ↓
91-D Geometric Feature Extraction
         ↓
Random Forest Classifier (300 trees)
         ↓
98.89% Test Accuracy
```

---

## 📊 Feature Engineering (91-D Vector)

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| Normalized coordinates | 63 | (x,y,z) per landmark, wrist-centered |
| Fingertip distances | 5 | Each fingertip to wrist |
| Finger extension | 5 | Tip to base distance |
| Inter-fingertip distances | 10 | Pairwise fingertip distances |
| Joint angles | 8 | Bend angles + spread + orientation |
| **Total** | **91** | |

---

## 📈 Training Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.89%** |
| Training Samples | 52,076 |
| Test Samples | 13,019 |
| Classes | 29 (A–Z + del + space + nothing) |
| Training Time | ~89 minutes (CPU only) |

---

## ⚙️ Configuration (`config.py`)

```python
ENABLE_TUNING = False    # Set True for hyperparameter search
N_ESTIMATORS  = 300      # Number of trees
CV_FOLDS      = 5        # Cross-validation folds
N_ITER_SEARCH = 30       # RandomizedSearchCV iterations
```

---

## 🚀 How to Train

**Step 1 — Install dependencies:**
```bash
pip install -r requirements_train.txt
```

**Step 2 — Add dataset:**
```
Place ASL dataset in:
asl_alphabet_train/asl_alphabet_train/
```

**Step 3 — Run training:**
```bash
python train.py
```

**Step 4 — Models saved to:**
```
models/rf_model_68.pkl
models/label_encoder.pkl
```

---

## ☁️ Training on Google Colab

```python
# Install dependencies
!pip install mediapipe==0.10.13 scikit-learn==1.6.1 opencv-python-headless numpy

# Clone repo
!git clone https://github.com/teejan7/sign-language-translator.git
%cd sign-language-translator

# Mount Drive to save models
from google.colab import drive
drive.mount('/content/drive')

# Train
!python train.py

# Save to Drive
import shutil, os
os.makedirs('/content/drive/MyDrive/asl_models', exist_ok=True)
shutil.copy('models/rf_model_68.pkl', '/content/drive/MyDrive/asl_models/rf_model_68.pkl')
shutil.copy('models/label_encoder.pkl', '/content/drive/MyDrive/asl_models/label_encoder.pkl')
```

---

## 🖥️ Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| CPU | Intel Core i3 or above |
| GPU | Not required ✅ |
| RAM | 8GB recommended |
| Storage | ~2GB for dataset |

---

## 📦 Dependencies

```
mediapipe==0.10.13
scikit-learn==1.6.1
opencv-python-headless
numpy
```

---
