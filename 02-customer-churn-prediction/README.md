# 02 — Customer Churn Prediction

A complete deep learning pipeline for predicting bank customer churn using PyTorch. Covers data preprocessing, handling class imbalance, model optimization, and deployment-ready export.

## Objective

Build a neural network that predicts whether a bank customer will leave (churn) based on their profile — demographics, account details, and activity. The dataset is imbalanced (~80% stayed, ~20% churned), making this a realistic binary classification problem.

## Dataset

- **Source:** Bank customer data (10,000 records)
- **Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target:** Exited (1 = Churned, 0 = Stayed)
- **Class distribution:** ~80/20 imbalance

## Pipeline Overview

1. **Data Cleaning** — Drop irrelevant columns (RowNumber, CustomerId, Surname), handle missing values
2. **Encoding** — OneHotEncoder with `drop='first'` for Geography and Gender via ColumnTransformer
3. **Scaling** — StandardScaler on numeric features only (one-hot columns stay as 0/1)
4. **Class Imbalance** — `pos_weight` in BCEWithLogitsLoss (3.87x penalty for missing churners)
5. **Training** — Batched training with DataLoader, Adam optimizer with L2 regularization
6. **Evaluation** — Classification report, confusion matrix, ROC-AUC curve
7. **Optimization** — Dynamic quantization comparison + ONNX export
8. **Deployment** — Save model weights, scaler, ONNX model, and metadata

## Model Architecture

```
ChurnModel(
  Linear(input → 64) → BatchNorm → ReLU → Dropout(0.3)
  Linear(64 → 32)    → BatchNorm → ReLU → Dropout(0.3)
  Linear(32 → 1)     → Raw logits (sigmoid applied in loss)
)
```

- **BatchNorm** for training stability
- **Dropout (0.3)** to prevent overfitting on 10K samples
- **BCEWithLogitsLoss** with class weights for numerical stability and imbalance handling
- **Adam optimizer** with weight decay (L2 regularization)

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~82% |
| AUC-ROC | ~0.85 |
| Churned Recall | ~65% |

> *Results may vary between runs. The key improvement is that the model actually identifies churners — without class weights, it predicts 0 churners despite ~80% accuracy.*


## Key Takeaways

- **Class imbalance kills naive models** — without `pos_weight`, the model predicts "Stayed" for everyone and still gets ~80% accuracy
- **Scaling matters for neural networks** — unscaled features like Balance (thousands) dominate over Age (tens), but one-hot columns should NOT be scaled
- **Accuracy is misleading** — for churn prediction, Recall on the churned class and AUC-ROC are far more important metrics
- **Quantization doesn't help small models** — dynamic quantization actually slowed inference by ~2x due to int8↔float32 conversion overhead; ONNX export is the better optimization for this model size
- **Save more than just weights** — the scaler and feature metadata are essential for correct predictions on new data

## Optimization Comparison

| Method | Inference (1000 runs) | Notes |
|--------|----------------------|-------|
| PyTorch (original) | ~0.5s | Baseline |
| Quantized (int8) | ~1.0s | Slower — overhead exceeds gains for small models |
| ONNX Runtime | Faster | Framework-agnostic, recommended for deployment |

## Deployment Artifacts

The notebook saves four files needed for production deployment:

- `churn_model.pth` — PyTorch weights (for retraining/fine-tuning)
- `churn_model.onnx` — ONNX model (for production inference without PyTorch)
- `scaler.pkl` — Fitted StandardScaler (must use same scaling on new data)
- `metadata.json` — Feature names, column order, threshold, accuracy

## How to Run

1. Open in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/deep-learning-pytorch/blob/main/02-customer-churn-prediction/notebooks/customer_churn_prediction.ipynb)

   > Replace `YOUR_USERNAME` with your GitHub username.

2. Upload `customer_data.csv` to Colab or mount Google Drive
3. Enable GPU (optional): **Runtime → Change runtime type → T4 GPU**
4. Run all cells: **Runtime → Run all** (`Ctrl+F9`)

## Project Structure

```
02-customer-churn-prediction/
├── README.md
├── notebooks/
│   └── customer_churn_prediction.ipynb
└── requirements.txt
```

## Requirements

```
torch>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
onnx>=1.14
onnxruntime>=1.15
```
