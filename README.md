# GCP Marker Localization and Shape Classification

## Skylark Drones Internship Assignment

### Author

**Sriperambuduru Pavithran**
B.Tech CSE with specialization in Artificial Intelligence and Data Science, IIIT Kottayam

GitHub: https://github.com/pavithrans05

---

# Problem Statement

The objective of this assignment is to build a computer vision system capable of:

1. Predicting the pixel coordinates **(x, y)** of the center of a Ground Control Point (GCP) marker.
2. Classifying the marker shape into one of the following categories:

   * Cross
   * Square
   * L-Shape

This is formulated as a **multi-task learning problem** consisting of:

* **Regression Task:** Predict GCP center coordinates.
* **Classification Task:** Predict marker shape.

---

# Dataset Overview

The provided dataset contains aerial drone imagery captured from multiple projects and locations.

## Dataset Statistics

* Total annotations: **1000**
* Valid samples used: **996**
* Missing shape labels: **4**
* Test images: **300**

## Shape Distribution

| Shape   | Count |
| ------- | ----: |
| Cross   |   177 |
| Square  |   328 |
| L-Shape |   491 |

The dataset exhibits class imbalance, which was addressed using weighted classification loss.

---

# Exploratory Data Analysis (EDA)

The following analyses were performed before model development:

* Dataset structure analysis
* Annotation verification
* Missing label analysis
* Shape distribution analysis
* Coordinate distribution analysis
* Image resolution analysis
* Sample visualization of GCP markers

Major observations:

* Most images have resolutions:

  * 4096 × 2730
  * 4096 × 3068
* GCP coordinates span nearly the entire image area.
* Only four samples lacked shape labels and were excluded.

---

# Methodology

## Data Preprocessing

For each image:

1. Image loaded using OpenCV.
2. Ground truth coordinates normalized to `[0,1]`.
3. Images resized to **512 × 512**.
4. Coordinates transformed accordingly.

---

## Data Augmentation

Training augmentations:

* Resize (512 × 512)
* Horizontal Flip
* Vertical Flip
* Random Rotation
* Random Brightness/Contrast
* Gaussian Blur

Validation data only uses resizing.

---

# Model Architecture

A **multi-task learning architecture** based on **EfficientNet-B0** was implemented.

## Backbone

* EfficientNet-B0 pretrained on ImageNet was used as the feature extraction backbone.

## Regression Head

Predicts normalized:

* x-coordinate
* y-coordinate

Architecture:

* Linear(1280 → 256)
* ReLU
* Dropout(0.2)
* Linear(256 → 2)
* Sigmoid

## Classification Head

Predicts marker shape:

* Cross
* Square
* L-Shape

Architecture:

* Linear(1280 → 256)
* ReLU
* Dropout(0.2)
* Linear(256 → 3)

---

# Training Strategy

## Train-Validation Split

* 80% Training
* 20% Validation
* Stratified split by marker shape

## Optimizer

* AdamW
* Learning Rate: 1e-3
* Weight Decay: 1e-4

## Scheduler

ReduceLROnPlateau:

* Factor: 0.5
* Patience: 2 epochs

## Early Stopping

Training stops if validation loss does not improve for 5 epochs.

---

# Loss Functions

## Regression Loss

Smooth L1 Loss

## Classification Loss

Weighted Cross Entropy Loss

Class weights:

* Cross: 1.8685
* Square: 1.0127
* L-Shape: 0.6769

Final multitask loss:

Loss = (5 × Regression Loss) + (1 × Classification Loss)

---

# Evaluation Metrics

The following metrics were used:

## Classification Metrics

* Accuracy
* Macro F1 Score

## Localization Metrics

* PCK@10
* PCK@25
* PCK@50

PCK (Percentage of Correct Keypoints) measures localization accuracy within pixel thresholds.

---

# Final Model Performance

Best Validation Loss: **0.1729**

Best Macro F1 Score: **1.0000**

Predictions Generated: **300 / 300**

---

# Assumptions

1. Coordinates provided in annotations are accurate.
2. Samples without shape labels were excluded.
3. Marker centers remain visible after augmentation.
4. Coordinate normalization preserves spatial relationships.

---

# Project Structure

```text
GCP_Assignment/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── transforms.py
│   ├── train.py
│   └── infer.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── GCP_Training_REG5.ipynb
│
├── weights/
│   ├── best_loss_reg5.pth
│   ├── best_f1_reg5.pth
│   └── training_history_reg5.csv
│
├── predictions.json
├── README.md
└── requirements.txt   
```

---

# Training Instructions

```bash
cd src
python train.py
```

---

# Inference Instructions

Place the trained model inside:

```text
weights/
```

Run inference:

```bash
cd src
python infer.py
```

The generated predictions will be saved as:

```text
predictions.json
```

---

# Model Weights

The trained model weights can be downloaded from the following Google Drive link:

https://drive.google.com/file/d/13V4z7FdHpA9VzKU2QJp3rf-eeWOnD50t/view?usp=sharing


---

# Reproducibility

Random seeds were fixed where applicable to ensure reproducible experiments. Training was performed on Google Colab using an NVIDIA Tesla T4 GPU.

---

# Conclusion

This project presents a robust multi-task learning framework for GCP localization and shape classification in aerial imagery. The final model achieves perfect classification performance while maintaining accurate spatial localization across diverse drone datasets.