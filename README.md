# 📡 Telecom Customer Churn Prediction — MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-D24939?logo=jenkins&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployed-326CE5?logo=kubernetes&logoColor=white)

**End-to-end MLOps pipeline for predicting telecom customer churn — dual-model inference (AdaBoost + XGBoost), Jenkins CI/CD, Docker containerisation, and Kubernetes deployment.**

</div>

---

## 🔍 Overview

This project transforms a telecom churn analysis notebook into a **production-grade MLOps system**. Two boosting models are trained in parallel and exposed through a Flask web app, shipped via a Jenkins shared-library CI/CD pipeline onto a Kubernetes cluster.

- **Dataset** — IBM Telco Customer Churn · 7,043 customers · 20 features · 26.5% churn rate
- **Problem** — Binary classification: will a customer churn?
- **Output** — Per-model churn label, probability score, risk level (High / Medium / Low), and a consensus card

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **AdaBoost** | 75.20% | 0.5236 | 0.7406 | 0.6135 | 0.8323 |
| **XGBoost** | 77.61% | 0.5603 | 0.7326 | 0.6350 | 0.8354 |

> XGBoost edges ahead on accuracy and ROC-AUC. AdaBoost leads on recall — it catches more actual churners, which matters most in a retention campaign context.

---

## 📁 Project Structure

```
MLOps_Churn/
│
├── src/                          # Core pipeline modules
│   ├── data_processing.py        # Cleaning, feature engineering, scaling
│   ├── model_training.py         # AdaBoost + XGBoost training & evaluation
│   ├── logger.py                 # Timestamped logging
│   └── custom_exception.py       # File/line-aware exception handling
│
├── artifacts/
│   ├── raw/                      # Source CSV
│   ├── processed/                # Scaled train/test splits + scaler + feature list
│   └── models/                   # Trained model pkl files + metrics.json
│
├── notebook/
│   ├── ML_Model_Building.ipynb   # Full modelling notebook
│   └── local ml building.ipynb   # End-to-end pipeline verification
|   |___Churn Analysis-EDA.ipynb  # EDA analysis
│
├── templates/ & static/          # Flask UI (dark telecom theme)
├── k8s/                          # Kubernetes deployment + service manifests
├── application.py                # Flask app — dual model inference
├── pipeline
|   |__ training_pipeline.py          # Pipeline entry point
|   
├── Dockerfile                    
├── Jenkinsfile                   # 5-stage CI/CD pipeline
└── requirements.txt
```

---

## ⚙️ Feature Engineering

Preprocessing mirrors the original notebook exactly — 34 final features.

| Step | What happens |
|---|---|
| **TotalCharges fix** | Coerced to numeric; 11 null rows dropped |
| **Tenure binning** | Raw tenure (0–72 months) slabbed into `1-12`, `13-24`, `25-36`, `37-48`, `49-60`, `61-72` via `pd.cut` |
| **Drop raw tenure** | Replaced entirely by `tenure_bin` |
| **One-hot encoding** | `get_dummies(drop_first=True)` on all categoricals — `tenure_bin_1-12` is the reference (all zeros) |
| **Scaling** | `StandardScaler` fitted on train only; same scaler applied at inference |

---

## 🏗️ Pipeline Architecture

```
Raw CSV
  └─► DataProcessing   →  cleaned + encoded + scaled artifacts
        └─► ModelTraining  →  AdaBoost pkl + XGBoost pkl + metrics.json
              └─► Flask App     →  dual prediction + risk scoring
                    └─► Docker   →  containerised image
                          └─► Jenkins  →  CI/CD on every push to main
                                └─► Kubernetes  →  2-replica live deployment
```

---

## 🚀 Quick Start

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Place the dataset**
```
artifacts/raw/data.csv
```

**3. Run the training pipeline**
```
python training_pipeline.py
```

**4. Launch the app**
```
python application.py  →  http://localhost:5000
```

---
##  Refere documentation.md for detailed explanation and library setup

## 🔄 CI/CD — Jenkins (5 stages)


| Stage | Action |
|---|---|
| Checkout | Pull latest from GitHub |
| Training Pipeline | Retrain both models from scratch |
| Build & Push Image | Docker build → push to Docker Hub |
| Install Kubectl | Configure cluster access |
| Deploy to Kubernetes | Rolling update via `kubectl apply` |

Every push to `main` triggers a full retrain and redeploy automatically.

---

## ☸️ Kubernetes

| Config | Value |
|---|---|
| Replicas | 2 |
| Image | `vel23/mlops-jenkins-churn-prediction:latest` |
| Memory limit | 512Mi |
| CPU limit | 500m |
| Service type | NodePort |
| External port | 30080 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | AdaBoost · XGBoost · scikit-learn |
| Web | Flask · Gunicorn |
| Tuning | Optuna · RandomizedSearchCV |
| Imbalance handling | SMOTE · SMOTEENN · ADASYN · `scale_pos_weight` |
| Containerisation | Docker |
| Orchestration | Kubernetes |
| CI/CD | Jenkins Shared Library |
| Cloud | GCP |



---

<div align="center">
  <sub>Built with ❤️ · MLOps · Jenkins · Kubernetes · AdaBoost · XGBoost</sub>
</div>
