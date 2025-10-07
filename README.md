# Facial Expression Recognition with Convolutional Neural Networks

## Overview
This repository presents a implementation of a Convolutional Neural Network (CNN) for automatic **facial expression recognition**. The model is trained on the **FER2013 dataset**, from which we took 5 class of emotions, and is designed to classify grayscale facial images into discrete emotion categories. The project is written in **Python** with **Pytorch**.

- **Objective:** Predict human emotions from facial images in real-time and static settings.
- **Research context:** Facial expression analysis is a key subfield in affective computing and human–computer interaction. This project builds a baseline model to explore its challenges (low-resolution data, class imbalance, intra-class variability).

---

## Dataset
**FER2013 (Facial Expression Recognition 2013), only a 5 class subset**
- Source: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- 35,887 images, 48×48 pixels, grayscale
- Labels: 5 emotions (anger, happiness, sadness, surprise, neutral)

---

## Model Architecture

The proposed CNN model is a **deep convolutional network** composed of four convolutional blocks followed by fully connected layers.

### Convolutional Blocks
Each block contains **two convolutional layers** with **Batch Normalization** and **ReLU activation**, followed by **MaxPooling** and **Dropout**:

| Block | Filters | Structure | Dropout |
|:------|:---------|:-----------|:---------|
| 1 | 64 | (Conv → BN → ReLU) ×2 + MaxPool | 0.25 |
| 2 | 128 | (Conv → BN → ReLU) ×2 + MaxPool | 0.25 |
| 3 | 256 | (Conv → BN → ReLU) ×2 + MaxPool | 0.35 |
| 4 | 512 | (Conv → BN → ReLU) ×2 + MaxPool | 0.40 |

Each convolution uses a **3×3 kernel with padding=1** to preserve spatial dimensions before pooling.


- Hidden layers use **ReLU** activation.  
- The final layer outputs **raw logits** (later passed through Softmax for emotion probabilities).  
- A **Dropout of 0.5** is applied between dense layers.

---

> **Input:** Grayscale face image (1 × 48 × 48)  
> **Output:** Probability distribution over 5 emotion classes.

---

### Fully Connected Head
After flattening, the feature map (of size `512 × 3 × 3`) is passed through **three dense layers** with dropout regularization:

<p align="center">
  <b>Figure 1 – CNN Model Architecture</b><br>
  <img src="CNN Model Architecture.png" width="70%" alt="CNN Model Architecture">
</p>

---

## Methodology
1. **Preprocessing:**
   - Grayscale conversion
   - Normalization with dataset mean and standard deviation
   - Data augmentation (random flips, small rotations) for robustness

2. **Training protocol:**
   - Loss: Cross-Entropy
   - Optimizer: Adam with weight decay

3. **Evaluation:**
   - Metrics: Accuracy, per-class accuracy
   - Training/validation curves to monitor generalization

---

## Results
- **Baseline accuracy:** 73 % (to be reported after experiments)

Key observations:
- Overfitting risk mitigated via dropout and augmentation
- Certain emotions (e.g., happy, neutral) are classified more reliably than others

---

## Real-Time Extension
A real-time pipeline has been implemented for webcam-based emotion recognition:
- Face detection: Haar cascades (OpenCV)
- Preprocessing: identical normalization as training
- Prediction: model inference on detected face ROI
- Visualization: live overlay of predicted emotion and probability bars

This demo illustrates the applicability of the trained model for human–machine interaction scenarios.

---

## Limitations and Future Work
- FER2013 is low-resolution and noisy → accuracy is capped
- Model performance is sensitive to lighting and head pose
- Future directions:
  - Use higher-quality datasets (RAF-DB, AffectNet)
  - Add more emotions classes

---


## References
1. Zhang, K., Zhang, Z., Li, Z., Qiao, Y. (2016). *Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks*. arXiv:1604.02878.
2. Kaggle FER2013 dataset: https://www.kaggle.com/datasets/msambare/fer2013.
3. Christian Białek,Andrzej Matiolański and Michał Grega (2023) An Efficient Approach to Face Emotion Recognition with Convolutional Neural Networks https://www.mdpi.com/2079-9292/12/12/2707.

