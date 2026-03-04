# 01 — CIFAR-10 CNN Comparison

Comparing two CNN architectures on the CIFAR-10 image classification dataset to demonstrate the impact of modern techniques like Batch Normalization and Dropout.

## Objective

Train a **BasicCNN** (classic tutorial architecture) and an **ImprovedCNN** (with BatchNorm + Dropout) side by side, then evaluate both with proper metrics beyond simple accuracy.

## Models

### BasicCNN
- 2 convolutional layers (6 and 16 filters, 5×5 kernels)
- 3 fully connected layers
- No regularization
- ~62K parameters

### ImprovedCNN
- 3 convolutional layers (32, 64, 128 filters, 3×3 kernels with padding)
- Batch Normalization after each conv layer
- Dropout (0.5) in the classifier
- ~295K parameters

## Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| BasicCNN | ~53% | Struggles with similar classes (cat/dog, deer/horse) |
| ImprovedCNN | ~78% | Significant improvement from BatchNorm + Dropout |

> *Results may vary slightly between runs. Train for more epochs or tune hyperparameters for better performance.*


## Key Takeaways

- **BatchNorm** stabilizes training and allows faster convergence
- **Dropout** reduces overfitting on the small CIFAR-10 dataset
- **3×3 kernels with padding** preserve spatial information better than 5×5 without padding
- **Proper evaluation** (precision, recall, F1, confusion matrix) reveals class-level weaknesses that overall accuracy hides

## How to Run

1. Open in Google Colab:
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Runtime → Run all** (`Ctrl+F9`)

## Project Structure

```
01-cifar10-cnn-comparison/
├── README.md
├── notebooks/
   └── cifar10_comparison.ipynb    # Complete pipeline

```

## Requirements

```
torch>=2.0
torchvision>=0.15
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
numpy>=1.24
```
