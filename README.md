# CKD Stage Prediction - Deep Learning Project

Deep learning system for Chronic Kidney Disease (CKD) stage prediction using CT scan segmentation and classification.

## 🎯 Project Overview

- **Segmentation**: U-Net model for kidney segmentation (Dice Score > 0.92)
- **Classification**: EfficientNet-B0 for CKD stage classification (4 stages)
- **Web Interface**: Flask-based UI for CT slice analysis

## 📁 Project Structure

```
ckdpred/
├── app.py                          # Main Flask application
├── show_progress.py                 # Training progress viewer
├── models_ckd/                     # Trained models
│   ├── unet_kidney_10k_30ep.pth
│   └── efficientnet_ckd_best_focal.pth
├── src/
│   ├── models/                     # Model architectures
│   ├── train/                      # Training scripts
│   │   └── train_ckd_classifier_focal.py  # Main training script
│   ├── preprocessing/              # Data preprocessing
│   ├── data/                       # Dataset classes
│   ├── inference/                  # Inference scripts
│   └── analysis/                   # Analysis tools
├── data/
│   ├── raw/                        # Raw CT slices and labels
│   ├── classification/             # ROI slices for classification
│   └── annotations/                # Segmentation masks
├── templates/                      # HTML templates
├── static/                         # CSS and results
└── logs/                           # Training logs

```

## 🚀 Quick Start

### 1. Run Web Application
```bash
python app.py
```
Access at: `http://127.0.0.1:5000`

### 2. Train Model
```bash
python src/train/train_ckd_classifier_focal.py
```

### 3. Monitor Training Progress
```bash
python show_progress.py
```

## 📊 Features

- **4 CKD Stages**: Well Preserved → Mild → Moderate → Advanced
- **Segmentation-led ROI**: U-Net extracts kidney regions
- **High Accuracy**: >96% validation accuracy
- **Professional UI**: Modern, accessible interface

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Flask
- OpenCV
- Albumentations
- CUDA-capable GPU (recommended)

## 📝 Notes

- Training uses Focal Loss to address class bias
- Models are saved automatically during training
- Latest training logs are kept in `logs/` directory

# Results And Screenshots
<img width="1919" height="971" alt="Screenshot 2026-05-15 154117" src="https://github.com/user-attachments/assets/f8bc2769-b7bf-47d7-8392-996abfa3b9fc" />
<img width="1910" height="905" alt="Screenshot 2026-05-15 154134" src="https://github.com/user-attachments/assets/4d79818a-beaf-4423-8eff-6a30e325c0df" />
<img width="1887" height="899" alt="Screenshot 2026-05-15 154211" src="https://github.com/user-attachments/assets/a5bddc08-9948-4169-9707-1f5aa0f9d319" />
<img width="817" height="567" alt="Screenshot 2026-03-14 210846" src="https://github.com/user-attachments/assets/f1d4e844-7469-473e-bc11-36930dd32448" />
<img width="932" height="897" alt="Screenshot 2026-03-14 210616" src="https://github.com/user-attachments/assets/2be74ff3-e2c3-4e39-93c2-2674929b2b60" />
<img width="591" height="525" alt="Screenshot 2026-03-14 210944" src="https://github.com/user-attachments/assets/ea596377-7174-4ff7-bed2-a6ee3cb096b7" />
<img width="1919" height="1079" alt="Screenshot 2026-03-19 010255" src="https://github.com/user-attachments/assets/3d6c06b2-254c-4553-b9fe-f99a7e7e959a" />






