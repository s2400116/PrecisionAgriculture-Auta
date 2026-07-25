# 🌿 Precision Agriculture: Plant Leaf Disease & Anomaly Detection

An end-to-end Computer Vision pipeline built with **YOLOv8** to identify, localize, and classify crop health issues from leaf imagery. Designed for real-time monitoring and integration with agricultural applications.

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg)](https://docs.ultralytics.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Release](https://img.shields.io/github/v/release/s2400116/PrecisionAgriculture-Auta?color=green)](https://github.com/s2400116/PrecisionAgriculture-Auta/releases/tag/v1.2.0)

---

## 🎯 Features & Target Classes

The model detects 3 primary agricultural anomalies across crop fields:
* 🦠 **Fungal Infection**: Visible spot patterns, blights, or mildew on leaf surfaces.
* 🐛 **Pest Damage**: Feeding marks, punctures, and tissue destruction caused by insects.
* 🍂 **Leaf Anomaly**: General discolouration, nutrient deficiency signs, or structural deformation.

---

## 📊 Model Performance & Benchmark Results

Trained on a dataset of **600 annotated images** with **1,665 instances** over **100 epochs** using **YOLOv8 Small (`yolov8s.pt`)** at **800 × 800** resolution.

### ⚙️ Training Environment & Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Model Architecture** | YOLOv8s (`yolov8s.pt`) |
| **Framework** | PyTorch `2.11.0+cu128` / Ultralytics `8.4.105` |
| **Hardware** | NVIDIA Tesla T4 (15 GB VRAM) |
| **Image Size (`imgsz`)** | 800 × 800 |
| **Batch Size** | 16 |
| **Optimizer** | AdamW (`lr0=0.001429`, `momentum=0.9`) |
| **Epochs** | 100 |
| **Training Time** | ~41.8 minutes (0.696 hrs) |
| **Parameters** | 11,126,745 (11.1M) |
| **GFLOPs** | 28.4 |

---

### 🏆 Validation Metrics (`best.pt`)

* **Validation Images:** 600
* **Total Instances:** 1,665

| Metric | Score |
| :--- | :--- |
| **Precision ($P$)** | **88.6%** (`0.886`) |
| **Recall ($R$)** | **84.5%** (`0.845`) |
| **$\text{mAP}_{50}$** | **92.1%** (`0.921`) |
| **$\text{mAP}_{50~95}$** | **57.2%** (`0.572`) |

---

### 🏷️ Per-Class Performance Breakdown

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | $\text{mAP}_{50}$ | $\text{mAP}_{50~95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **600** | **1665** | **0.886** | **0.845** | **0.921** | **0.572** |
| 🐛 Pest Damage | 127 | 208 | **0.914** | **0.933** | **0.972** | **0.644** |
| 🍂 Leaf Anomaly | 285 | 826 | **0.897** | **0.812** | **0.905** | **0.546** |
| 🦠 Fungal Infection | 268 | 631 | **0.846** | **0.791** | **0.887** | **0.525** |

---

### ⚡ Speed Benchmark (Tesla T4)

* **Pre-process:** 0.3 ms
* **Inference:** 7.3 ms
* **Post-process:** 3.0 ms
* **Total Latency:** ~10.6 ms (~94 FPS)

---

## 📦 Pre-trained Model Weights

Pre-trained model weights (`best.pt`) are available directly from the repository release page:

📥 **[Download best.pt (Release v1.0.0)](https://github.com/s2400116/PrecisionAgriculture-Auta/releases/tag/v1.0.0)**

---

## 🖼️ Evaluation & Outputs

<div align="center">
  <img src="assets/confusion_matrix.png" width="45%" alt="Confusion Matrix" />
  <img src="assets/results.png" width="45%" alt="Training Metrics" />
</div>

<br/>

<div align="center">
  <img src="assets/demo_preview.jpg" width="90%" alt="Detection Preview" />
  <p><i>Sample model predictions on validation batch.</i></p>
</div>

---

## 📁 Repository Structure

```text
PrecisionAgriculture-Auta/
│
├── assets/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── demo_preview.jpg
│
├── config/
│   └── data.yaml
│
├── notebook/
│   └── PrecisionAgriculture.ipynb
│
├── src/
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md
