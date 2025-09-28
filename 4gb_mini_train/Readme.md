
#  YOLOv8-Style Object Detection Model

This project implements a YOLOv8-style object detection pipeline in PyTorch using a custom dataset. The goal is to detect and classify objects with high precision and recall, optimized for real-world scale datasets and hardware limitations (e.g., Kaggle GPU runtime).

---

## 📊 Training Summary

- **Dataset Size:** (Dataset is originally COCO dataset)
dataset can be found here :
https://www.kaggle.com/datasets/anitatasnimproma/coc-yolo

  - `train/`: 19,400 images  
  - `val/`: 4,866 images  
  - **Total**: ~24,266 images (~4 GB)

- **Training Details:**
  - **Model**: YOLOv8-style custom architecture
  - **Framework**: PyTorch
  - **Augmentations**: Mosaic, MixUp, HSV, Albumentations, Random Perspective
  - **Optimizer**: AdamW
  - **Loss Function**: CIoU + DFL + BCE (YOLOv8-style)
  - **Learning Rate Scheduler**: Cosine Annealing
  - **AMP (Automatic Mixed Precision)**: Enabled
  - **EMA (Exponential Moving Average)**: Enabled
  - **Device**: Kaggle GPU (Tesla T4, 16 GB VRAM)
  - **Epochs**: Target = 100, Completed = 71 (interrupted due to Kaggle timeout)
  - **Batch Size**: Configurable

### 📈 Validation Performance (Up to Epoch 71)

| Epoch | Loss  | Precision | Recall | mAP@0.5 |
|-------|-------|-----------|--------|---------|
| 69    | 4.420 | 0.549     | 0.416  | 0.298   |
| 70    | 4.426 | 0.548     | 0.418  | 0.299   |
| 71    | 4.401 | ✱         | ✱      | ✱ *(crashed during mAP eval)* |

> ⚠️ **Note:** Training was interrupted during epoch 71's mAP evaluation due to the Kaggle 12-hour GPU time limit.

---

##  Dataset Optimization Rationale

Originally, the dataset was significantly larger, but it was reduced to ~4 GB for the following reasons:

- **Kaggle GPU Time Limit:** Kaggle imposes a 12-hour cap on GPU sessions. A smaller dataset enables completion of more training epochs within this limit.

- **Faster I/O and Training:** Reducing dataset size lowered disk I/O overhead and improved training speed—critical in cloud-based environments.

- **Efficient Iteration:** Smaller datasets allowed for faster experimentation with various architectures and hyperparameters.

- **Performance Saturation:** Initial tests indicated that mAP gains diminished beyond ~20k images, prompting a focus on quality over quantity.


