# Head Detection on MPII using RT-DETR and Custom Transformer

## Overview

This project addresses the problem of **human head detection** in images using the MPII Human Pose dataset.

The objective is to compare different approaches to object detection, ranging from traditional convolutional methods to modern Transformer-based architectures.

The project includes:

* A CNN-based baseline (ResNet)
* A state-of-the-art model (RT-DETR)
* A custom Transformer-based implementation (SimpleRTDETR)

---

## Motivation

Traditional CNN-based approaches struggle to capture global spatial relationships in images, often leading to poor localization performance in multi-object scenarios.

This project explores how **Transformer-based architectures** can overcome these limitations through global attention mechanisms.

---

## Dataset

* **Dataset:** MPII Human Pose Dataset
* The dataset was preprocessed to extract **head bounding boxes**
* Non-relevant annotations (e.g., bystanders without keypoints) were filtered out

A custom preprocessing pipeline was implemented to:

* Parse `.mat` annotation files
* Generate bounding boxes
* Convert data into YOLO-compatible format

---

## Approaches

### 1. CNN Baseline (ResNet)

* ResNet18 used as feature extractor
* Bounding box regression using MSE loss
* Limitation: model collapses to average predictions in multi-object scenarios

---

### 2. RT-DETR (Ultralytics)

* State-of-the-art real-time detection transformer
* Used both:

  * Pretrained (fine-tuning)
  * Training from scratch

Provides an upper bound for performance.

---

### 3. Custom Model — SimpleRTDETR

A simplified implementation of RT-DETR designed to better understand the internal mechanics of Transformer-based detection.

**Architecture:**

* Backbone: ResNet18
* Transformer: Encoder-Decoder (PyTorch `nn.Transformer`)
* Learnable object queries
* MLP heads for classification and bounding box prediction

---

## Results

| Model                 | mAP@50 | mAP@50-95 |
| --------------------- | ------ | --------- |
| ResNet (baseline)     | 0.536  | 0.299     |
| SimpleRTDETR (custom) | 0.7244 | 0.3645    |
| RT-DETR (fine-tuned)  | >0.93  | ~0.58     |

The custom implementation achieves competitive performance, validating the effectiveness of attention mechanisms for object detection.

---

## Key Insights

* CNN-based regression fails in multi-object settings due to averaging effects
* Transformer-based models successfully learn spatial relationships
* Even simplified architectures can achieve strong performance

---

## Project Structure

```bash
src/            # Models and training logic
notebooks/      # Experiments and visualization
report/         # Full academic report
```

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare dataset:

* Download MPII dataset
* Run preprocessing script

3. Train models:

```bash
python train_resnet.py
python train_rtdetr.py
```

---

## Report

The full academic report is available in:

```
report/informe_aarn.pdf
```

---

## Notes

* Model weights are not included due to size constraints
* Training was performed on a local GPU environment
* Code is adapted from a research/academic context
