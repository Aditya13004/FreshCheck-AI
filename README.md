# FreshCheck AI 🍃
**Enterprise-Grade Agricultural Quality Control & Analytics System**

![FreshCheck Dashboard](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal)

FreshCheck AI is a high-performance, full-stack application designed to automate the visual inspection of agricultural produce. Utilizing a Multi-Task Learning Convolutional Neural Network, it simultaneously classifies produce types and evaluates freshness across 17 different fruit and vegetable categories.

## 🚀 Key Features
* **Multi-Task Deep Learning:** Powered by a dual-head MobileNetV2 architecture, processing both classification and condition assessment simultaneously.
* **Explainable AI (XAI):** Mathematically extracts deep neural activations to generate a Class-Agnostic Heatmap, allowing users to visually inspect AI attention maps in real-time.
* **Out-of-Distribution (OOD) Anomaly Detection:** Implements strict probability thresholds to automatically intercept and flag unknown objects, preventing AI hallucinations.
* **Live Edge Inference:** Streams live webcam data directly into the backend via HTML5 Canvas binary blob conversion.
* **Multi-Angle Batch Verification:** Supports simultaneous batch uploads, aggregating results with a strict "Any Rotten = All Rotten" agricultural standard.
* **Modern Architecture:** Built on a lightning-fast FastAPI REST backend decoupled from a vanilla JS/CSS glassmorphism dashboard.

## 🛠 Tech Stack
* **Machine Learning:** TensorFlow 2.x, Keras, OpenCV, NumPy
* **Backend API:** FastAPI, Uvicorn, Python-Multipart
* **Frontend UI:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Web Speech API

## 🧠 Model Architecture
The AI was trained on a dynamically merged dataset of over 30,000 images. The training pipeline included:
1. Binary `Magic Byte` verification to sanitize corrupted dataset images.
2. Transfer learning via a frozen MobileNetV2 backbone.
3. Bifurcated dense heads optimized with categorical crossentropy (fruit classification) and binary crossentropy (freshness grading).

## ⚙️ Local Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FreshCheck-AI.git
   ```
2. Install the optimized dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   python server.py
   ```
4. Access the dashboard at `http://localhost:8000`
