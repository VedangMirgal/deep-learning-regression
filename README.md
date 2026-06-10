Here's a professional GitHub-style README for your **Deep Learning Regression** project.

---

# Deep Learning Architectures for Productivity Prediction

## Overview

This project explores and compares multiple deep learning architectures for predicting organizational productivity changes based on workforce, adoption, and training-related features. The objective is to evaluate how different neural network designs perform on structured tabular data and identify the most effective architecture for regression tasks.

Five deep learning models were implemented and benchmarked:

* Deep Artificial Neural Network (ANN)
* 1D Convolutional Neural Network (CNN)
* Autoencoder-Based Regressor
* Residual Neural Network (ResNet-inspired ANN)
* Variational Autoencoder (VAE) Regressor

The models were trained to predict **Productivity Change (%)** using company-level information such as industry, country, AI tool adoption, workforce impact, and training investments.

---

## Problem Statement

Organizations adopting Generative AI technologies often experience varying productivity outcomes.

Given organizational attributes such as:

* Industry
* Country
* AI Tool Adopted
* Adoption Year
* Employees Impacted
* New Roles Created
* Training Hours Provided

the task is to predict:

```text
Productivity Change (%)
```

This is formulated as a supervised regression problem.

---

## Dataset

The dataset contains organizational information related to Generative AI adoption.

### Features

| Feature                 | Description                   |
| ----------------------- | ----------------------------- |
| Company Name            | Organization identifier       |
| Industry                | Industry sector               |
| Country                 | Country of operation          |
| GenAI Tool              | AI tool adopted               |
| Adoption Year           | Year of AI adoption           |
| Employees Impacted      | Number of employees affected  |
| New Roles Created       | Number of new roles generated |
| Training Hours Provided | Employee training investment  |

### Target Variable

```text
Productivity Change (%)
```

---

## Data Preprocessing

The preprocessing pipeline includes:

### Categorical Encoding

Categorical features such as:

* Industry
* Country
* GenAI Tool

were converted into numerical representations.

### Feature Scaling

Numerical variables were normalized to improve neural network training stability.

### Train-Test Split

The dataset was divided into:

```text
80% Training
20% Testing
```

to evaluate generalization performance.

---

# Models Implemented

## 1. Deep Artificial Neural Network (ANN)

A fully connected feedforward neural network serving as the baseline model.

### Architecture

```text
Input
 ↓
Dense (128)
 ↓
BatchNorm
 ↓
ReLU
 ↓
Dropout
 ↓
Dense (64)
 ↓
ReLU
 ↓
Dense (1)
```

### Motivation

* Captures nonlinear feature interactions
* Strong baseline for tabular regression tasks

---

## 2. Convolutional Neural Network (CNN)

A 1D convolutional architecture adapted for tabular data.

### Architecture

```text
Input Features
 ↓
1D Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
Fully Connected Layer
 ↓
Output
```

### Motivation

* Learns local feature patterns
* Reduces parameter count
* Captures neighboring feature interactions

---

## 3. Autoencoder Regressor

An encoder-based architecture that learns compressed feature representations before prediction.

### Architecture

```text
Input
 ↓
Encoder
 ↓
Latent Representation
 ↓
Regression Head
 ↓
Output
```

### Motivation

* Reduces noise
* Learns compact representations
* Performs dimensionality reduction automatically

---

## 4. Residual ANN (Best Performing Model)

A ResNet-inspired architecture with skip connections.

### Architecture

```text
Input
 ↓
Dense Layer
 ↓
Dense Layer
 ↓
Residual Connection
 ↓
Output Layer
```

### Motivation

* Improves gradient flow
* Enables deeper learning
* Learns corrective feature mappings

---

## 5. Variational Autoencoder (VAE) Regressor

A probabilistic latent-space model that learns feature distributions.

### Architecture

```text
Input
 ↓
Encoder
 ↓
Mean + Variance
 ↓
Latent Sampling
 ↓
Regression Head
 ↓
Output
```

### Motivation

* Learns smooth latent representations
* Better uncertainty modeling
* Improved generalization in complex datasets

---

# Training Configuration

### Loss Function

```python
MSELoss()
```

### Optimizer

```python
Adam
```

### Evaluation Metrics

* Mean Absolute Error (MAE)
* R² Score

---

# Results

| Model                 | MAE      | R² Score |
| --------------------- | -------- | -------- |
| Deep ANN              | 0.42     | 0.70     |
| CNN                   | 0.37     | 0.79     |
| Autoencoder Regressor | 0.37     | 0.77     |
| Residual ANN          | **0.34** | **0.79** |
| VAE Regressor         | 0.42     | 0.74     |

---

# Best Model

### Residual ANN

```text
MAE = 0.34
R² = 0.79
```

The Residual ANN achieved the strongest overall performance, providing the lowest prediction error while explaining approximately 79% of the variance in productivity outcomes.

The residual connections improved learning stability and enabled the model to capture complex nonlinear relationships more effectively than the baseline ANN.

---

# Visualization

The project includes:

### Actual vs Predicted Scatter Plot

Evaluates prediction quality by comparing model outputs against ground truth values.

### Model Comparison Plot

Visual comparison of predictions generated by all five architectures.

### Performance Metrics

Comparison using:

* MAE
* R² Score

for all models.

---

# Key Findings

* Deep learning models can effectively model productivity outcomes from organizational AI adoption data.
* Residual Networks achieved the best balance between accuracy and generalization.
* CNNs performed surprisingly well on structured tabular data.
* Autoencoder-based approaches successfully reduced feature redundancy.
* Increased architectural complexity (VAE) did not necessarily result in better predictive performance.

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib

---

# Future Improvements

* Hyperparameter optimization
* Ensemble learning
* Transformer-based tabular architectures
* Feature importance analysis using SHAP
* Uncertainty-aware regression models
* Cross-validation benchmarking

---

## Author

**Vedang Mirgal**

Deep Learning exploration and comparative study of neural architectures for tabular regression problems.
