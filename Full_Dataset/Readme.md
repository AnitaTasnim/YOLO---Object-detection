# YOLOv8 Training Progress on Full COCO Dataset

I initiated training of the YOLOv8 model on the full COCO dataset (~20GB) using the Kaggle platform. The setup was successfully configured, and the model began training as expected. However, after completing 4 epochs, the training process was unexpectedly terminated—likely due to Kaggle's free-tier session time or resource limits (e.g., 9-hour runtime cap or memory constraints).

##  Dataset Preparation

We used the original COCO 2017 dataset, which includes:

- `train2017`: ~118,000 images
- `val2017`: ~5,000 images
- JSON annotations for both splits

### Steps Followed:

1. **Environment Setup**
    ```bash
    pip install -q scikit-learn
    ```

2. **Folder Structure Created**
    ```
    COCO/
    ├── images/
    │   ├── train2017/
    │   └── val2017/
    └── labels/
        ├── train2017/
        └── val2017/
    ```

3. **Dataset Downloaded and Extracted**
    ```bash
    wget -O train2017.zip http://images.cocodataset.org/zips/train2017.zip
    wget -O val2017.zip http://images.cocodataset.org/zips/val2017.zip
    wget -O annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    unzip train2017.zip -d COCO/images
    unzip val2017.zip -d COCO/images
    unzip annotations_trainval2017.zip -d COCO
    ```

4. **Annotations Converted to YOLO Format**
    - Loaded `instances_train2017.json` and `instances_val2017.json`
    - Converted bounding boxes to `[class_id, x_center, y_center, width, height]` format
    - Filtered out invalid or zero-area boxes

5. **Labels and Image File Alignment**
    - Each `.jpg` image had a matching `.txt` label file
    - Skipped missing entries or malformed annotations

6. **Train/Val File Lists Generated**
    ```python
    train_images = sorted(glob('COCO/images/train2017/*.jpg'))
    val_images = sorted(glob('COCO/images/val2017/*.jpg'))

    with open('train2017.txt', 'w') as f:
        f.write('\n'.join(train_images))
    with open('val2017.txt', 'w') as f:
        f.write('\n'.join(val_images))
    ```

---

## Training Overview

- **Model:** YOLOv8
- **Dataset Size:** ~20GB (COCO train + val)
- **Environment:** Google Colab Notebook (Free GPU)
- **Session Limitations:** ~5 hours runtime, limited VRAM (e.g., 16GB for T4)

### Training Behavior:
- Training initiated correctly
- Data loading worked as expected
- Model ran successfully for **4 epochs**
- Terminated due to **GPU out-of-memory (OOM)** followed by **runtime disconnect** when memory optimizations were attempted



## Training Output

The process was forcefully interrupted after 4 epochs due to Colab GPU runtime reset.  
Below is the training progress before disconnection:

| Epoch | VRAM   | Loss   | Precision | Recall  | mAP@0.5 |
|-------|--------|--------|-----------|---------|---------|
| 1/500 | 8.46GB | 9.463  | 0.293     | 0.0537  | 0.0138  |
| 2/500 | 8.49GB | 6.703  | 0.229     | 0.1400  | 0.0568  |
| 3/500 | 8.50GB | 6.111  | 0.253     | 0.1930  | 0.0915  |
| 4/500 | 8.50GB | 5.736  | —         | —       | —       |

> _(Incomplete due to disconnection during epoch 4)_


---

## Output Summary

- 4 epochs completed before failure
- Model checkpoint saved before termination (if manually handled)
- Dataset pipeline confirmed working for large-scale training


