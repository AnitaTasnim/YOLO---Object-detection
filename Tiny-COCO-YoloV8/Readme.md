# YOLOv8 Object Detection - PyTorch Reimplementation

This is a reimplementation of the YOLOv8 object detection pipeline using PyTorch. The project follows the original repository [jahongir7174/YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt).

---

##  Project Overview

The objective of this project was to train and evaluate the YOLOv8 model using a custom pipeline in PyTorch, with training scripts, configuration options, and dataset integration modeled after the Ultralytics ecosystem.

---

## ⚠️ Constraints and Workaround

Due to hardware and runtime limitations on Google Colab and Kaggle, attempting to train on the full COCO 2017 dataset (118k+ images) resulted in `OutOfMemoryError` and `RuntimeError` crashes.

### Workaround: Use of COCO128 (Tiny COCO)

To validate the correctness of the codebase and training pipeline, I utilized the **COCO128 dataset** — a lightweight, 128-image subset of COCO 2017. This dataset is commonly used for debugging and sanity checks.

> **COCO128 Info**  
> Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco128.yaml  
> Size: 7MB, 128 images  
> Structure: Similar to full COCO (with `images/train2017`, `labels/train2017`)

---

##  Training Configuration

- **Model**: YOLOv8-nano (or smallest variant available in the repo)
- **Epochs**: 500
- **Dataset**: COCO128
- **Environment**: Google Colab / Kaggle (free-tier runtime)
- **Command**: `bash main.sh 1 --train`

---

## 📊 Results

### Training Set (COCO128)

| Metric     | Value |
|------------|-------|
| Precision  | 0.811 |
| Recall     | 0.682 |
| mAP        | 0.566 |

### Test Set (COCO128)

| Metric     | Value |
|------------|-------|
| Precision  | 0.835 |
| Recall     | 0.677 |
| mAP        | 0.583 |

These results demonstrate that the model and pipeline are functioning properly, achieving reasonable performance even on a small dataset.

---

## 📁 Table of Contents

| Module / File | Description |
|---------------|-------------|
| `args/`       | Contains argument parsing and configuration utilities for training/testing. |
| `dataset/`    | Custom Dataset classes, transforms, and dataloaders for COCO/COCO128. |
| `util/`       | Utility scripts including loss functions, metric computations, EMA, and logging. |
| `nets/`       | YOLOv8 architecture implementation (backbone, head, decoders). |
| `main.py`     | Entry point script for training and testing the model. Includes parser and launcher logic. |

---

## 🔧 Repository Structure Summary

- `main.py`: Contains training and testing logic
- `main.sh`: Shell script to launch training with multi-GPU support
- `models/`: Contains model architecture files
- `utils/`: Utilities for dataset loading, metrics, EMA, etc.
- `datasets/COCO128`: Dataset directory with COCO-like structure (used for this run)

---

##  Future Work

With access to a machine with higher memory and longer runtime, the same pipeline can be used to train on the full COCO 2017 dataset:

```bash
# Example full dataset training (after setting paths in main.py)
bash main.sh 1 --train
