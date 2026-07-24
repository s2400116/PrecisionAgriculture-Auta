# 🌿 Precision Agriculture: Plant Leaf Disease & Anomaly Detection

An end-to-end Computer Vision pipeline built with **YOLOv8** to identify, localize, and classify crop health issues from leaf imagery. Designed for real-time monitoring and integration with agricultural applications.

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg)](https://docs.ultralytics.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Release](https://img.shields.io/github/v/release/s2400116/PrecisionAgriculture-Auta?color=green)](https://github.com/s2400116/PrecisionAgriculture-Auta/releases/tag/v1.1.0)

---

## 🎯 Features & Target Classes

The model detects 3 primary agricultural anomalies across crop fields:
* 🦠 **Fungal Infection**: Visible spot patterns, blights, or mildew on leaf surfaces.
* 🐛 **Pest Damage**: Feeding marks, punctures, and tissue destruction caused by insects.
* 🍂 **Leaf Anomaly**: General discolouration, nutrient deficiency signs, or structural deformation.

---

## 📊 Model Performance & Benchmark Results

Trained on a dataset of **450 annotated images** with **1,099 instances** over **100 epochs** using **YOLOv8 Small (`yolov8s.pt`)** at **800 × 800** resolution.

### ⚙️ Training Environment & Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Model Architecture** | YOLOv8s (`yolov8s.pt`) |
| **Framework** | PyTorch `2.11.0+cu128` / Ultralytics `8.4.104` |
| **Hardware** | NVIDIA Tesla T4 (15 GB VRAM) |
| **Image Size (`imgsz`)** | 800 × 800 |
| **Batch Size** | 16 |
| **Optimizer** | AdamW (`lr0=0.001429`, `momentum=0.9`) |
| **Epochs** | 100 |
| **Training Time** | ~30.8 minutes (0.513 hrs) |
| **Parameters** | 11,126,745 (11.1M) |
| **GFLOPs** | 28.4 |

---

### 🏆 Validation Metrics (`best.pt`)

* **Validation Images:** 450
* **Total Instances:** 1,099

| Metric | Score |
| :--- | :--- |
| **Precision ($P$)** | **85.8%** (`0.858`) |
| **Recall ($R$)** | **85.1%** (`0.851`) |
| **$\text{mAP}_{50}$** | **92.5%** (`0.925`) |
| **$\text{mAP}_{50~95}$** | **60.1%** (`0.601`) |

---

### 🏷️ Per-Class Performance Breakdown

| Class | Images | Instances | Precision ($P$) | Recall ($R$) | $\text{mAP}_{50}$ | $\text{mAP}_{50~95}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **450** | **1099** | **0.858** | **0.851** | **0.925** | **0.601** |
| 🐛 Pest Damage | 107 | 161 | **0.911** | **0.895** | **0.963** | **0.640** |
| 🍂 Leaf Anomaly | 214 | 515 | **0.832** | **0.843** | **0.913** | **0.596** |
| 🦠 Fungal Infection | 198 | 423 | **0.831** | **0.813** | **0.901** | **0.568** |

---

### ⚡ Speed Benchmark (Tesla T4)

* **Pre-process:** 0.4 ms
* **Inference:** 7.4 ms
* **Post-process:** 4.5 ms
* **Total Latency:** ~12.3 ms (~81 FPS)

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
