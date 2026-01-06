# 🦴 Fracture Detector – Bone Fracture Classification from X-ray Images

This project focuses on the **automatic detection of bone fractures from X-ray images**
using **deep learning techniques**.  
It implements a **binary classification system** that predicts whether an X-ray image
shows a **fractured** or **non-fractured** bone.

The project is designed as a **research and experimentation pipeline**, combining
model training in Jupyter notebooks with a runnable inference script and Docker-based deployment.

---

## 📌 Project Overview

Bone fracture diagnosis is a critical task in medical imaging and emergency care.
This project leverages **Convolutional Neural Networks (CNNs)** to learn visual patterns
associated with fractures from X-ray images and provide fast, automated predictions.

The system aims to assist clinicians by acting as a **decision-support tool**,
not as a replacement for professional medical judgment.

---

## 📌 Dataset Information

**Fracture Multi-Region X-Ray Data** — sourced from Kaggle:

🔗 https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

---

## 🧠 Objectives

- Automatically classify X-ray images as **Fractured** or **Not Fractured**
- Apply deep learning techniques to medical imaging data
- Achieve high classification accuracy on validation data
- Provide a reproducible and deployable ML pipeline
## 🗂️ Repository Structure

Fracture_Detector/
│
├── Bone_Fracture_Classifier.ipynb # Model training and experimentation
├── main.py # Inference script
├── Dockerfile # Docker image definition
├── docker-compose.yml # Container orchestration
├── README.md # Project documentation


---

## 📘 File Descriptions

### `Bone_Fracture_Classifier.ipynb`
- Loads and preprocesses X-ray images
- Defines and trains a CNN-based fracture classifier
- Evaluates model performance (accuracy, loss)
- Serves as the main research and experimentation notebook

### `main.py`
- Entry point for running model inference
- Loads the trained model
- Preprocesses an input X-ray image
- Outputs the predicted class (fractured / not fractured)

### `Dockerfile`
- Builds a Docker image containing all dependencies
- Ensures reproducible execution across environments

### `docker-compose.yml`
- Simplifies running the project using Docker
- Automates container setup and execution

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **NumPy**
- **OpenCV**
- **Jupyter Notebook**
- **Docker**

---

## 🧪 Model Workflow
X-ray Image
↓
Preprocessing
↓
CNN Feature Extraction
↓
Binary Classification
↓
Fractured / Not Fractured


---

## 📊 Results

- The trained model achieves **high validation accuracy** (~98%)
- Demonstrates strong performance on fracture vs non-fracture classification
- Confirms the effectiveness of CNNs for medical X-ray analysis

---

## 🚀 Running the Project

### ▶ Run inference locally
```bash
python main.py

🐳 Run using Docker

docker-compose up --build
