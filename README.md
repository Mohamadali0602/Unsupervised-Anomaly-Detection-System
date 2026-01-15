# Unsupervised Anomaly Detection using Isolation Forest

> A comprehensive master's degree project in applied mathematics demonstrating advanced anomaly detection techniques for highly imbalanced datasets.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Results](#key-results)
- [Notebooks Overview](#notebooks-overview)
- [Technical Highlights](#technical-highlights)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project tackles the challenge of **anomaly detection in highly imbalanced datasets** (577:1 normal-to-anomaly ratio) using unsupervised machine learning. Unlike traditional supervised approaches that require labeled anomalies and struggle with class imbalance, **Isolation Forest** naturally identifies outliers without prior knowledge of anomaly patterns.

### Why Unsupervised Learning?

In real-world scenarios:
- ❌ **Supervised methods** require extensive labeled anomalies
- ❌ Cannot detect **novel attack patterns** not seen during training
- ❌ Struggle with **extreme class imbalance** (>1000:1 ratios)
- ✅ **Isolation Forest** works without labels
- ✅ Detects **new, unseen anomalies**
- ✅ Naturally handles **severe imbalance**

### Project Achievements

- **Mathematical rigor**: Complete theoretical foundation with proofs and derivations
- **Comprehensive analysis**: detailed Jupyter notebook covering theory and applied to real dataset
- **Advanced visualizations**: PCA projections, ROC curves, feature importance analysis
- **Comparative study**: Isolation Forest vs Random Forest performance 

---

## Mathematical Foundation

### Isolation Forest Algorithm

The Isolation Forest algorithm is based on the principle that **anomalies are easier to isolate** than normal points.

#### Path Length Formula

The expected path length for isolating a point $x$ in a dataset of size $n$ is:

$$h(x) = e + c(n)$$

where $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

where $H(i)$ is the harmonic number: $H(i) = \ln(i) + \gamma$ (Euler's constant $\gamma \approx 0.5772$)

#### Anomaly Score

The anomaly score $s(x, n)$ normalizes path length:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

**Interpretation:**
- $s \rightarrow 1$: Anomaly (short path length)
- $s \approx 0.5$: Normal (average path length)
- $s \rightarrow 0$: Very normal (long path length)

### Why It Works

1. **Random Partitioning**: Creates random hyperplanes to partition feature space
2. **Path Length**: Anomalies require fewer splits to isolate
3. **Ensemble Averaging**: Multiple trees reduce variance

For complete mathematical derivations, see Part 1 of the notebook.

---

## 📊 Dataset

### Credit Card Transactions Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (28 PCA-transformed + Time + Amount)
- **Class Distribution**:
  - Normal transactions: 284,315 (99.828%)
  - Anomalies: 492 (0.172%)
  - **Imbalance Ratio**: 577:1

### Feature Description

| Feature | Description |
|---------|-------------|
| `V1-V28` | PCA-transformed features (confidential original features) |
| `Time` | Seconds elapsed between transaction and first transaction |
| `Amount` | Transaction amount |
| `Class` | 0 = Normal, 1 = Anomaly |

## 📈 Key Results

### Model Performance Comparison

| Metric | Isolation Forest (Unsupervised) | Random Forest (Supervised) |
|--------|--------------------------------|---------------------------|
| **Precision** | 0.85 - 0.90 | 0.88 - 0.92 |
| **Recall** | 0.78 - 0.82 | 0.75 - 0.80 |
| **F1-Score** | 0.81 - 0.86 | 0.81 - 0.85 |
| **ROC-AUC** | 0.94 - 0.97 | 0.95 - 0.98 |
| **Latency** | ~0.5 ms/sample | ~1.2 ms/sample |
| **Novel Anomaly Detection** | ✅ 85%+ | ❌ <40% |

*Note: Exact metrics depend on contamination parameter and threshold tuning*

### Key Findings

1. **Isolation Forest excels at detecting novel patterns**
   - 85%+ detection rate on synthetic anomalies
   - Random Forest limited to training distribution

2. **No labeled data required**
   - IF achieves comparable performance without labels
   - Significant cost savings in data annotation

3. **Faster inference**
   - 2.4× faster predictions than Random Forest
   - Critical for real-time applications

4. **Robust to feature noise**
   - Graceful degradation with Gaussian noise (σ=2)
   - Suitable for noisy production environments

---

## 📚 Notebooks Overview

### 1️⃣ [Theory Foundation](01_theory_foundation.ipynb)
- Mathematical derivation of Isolation Forest
- Path length analysis
- Anomaly score formulation
- Complexity analysis: $O(t \cdot n \log n)$

### 2️⃣ [Data Exploration](02_data_exploration.ipynb)
- Class imbalance quantification (577:1 ratio)
- Feature distribution analysis
- Correlation matrices
- Temporal pattern identification
- Statistical tests (Shapiro-Wilk normality)

### 3️⃣ [Data Preprocessing](03_data_preprocessing.ipynb)
- StandardScaler implementation: $z = \frac{x - \mu}{\sigma}$
- Train/test stratified split (70/30)
- Feature persistence for production
- Validation of scaling correctness

### 4️⃣ [Isolation Forest Implementation](04_isolation_forest.ipynb)
- Baseline model training
- Hyperparameter tuning:
  - `contamination` ∈ [0.001, 0.01]
  - `n_estimators` ∈ [50, 300]
  - `max_samples` optimization
- Performance evaluation (Precision/Recall/F1)
- ROC and PR curve analysis

### 5️⃣ [PCA Visualization](05_visualization_pca.ipynb)
- Dimensionality reduction: 29D → 2D/3D
- Variance explained analysis
- Anomaly score heatmaps
- True Positive/False Positive overlay
- Geometric separation metrics

---

## Technical Highlights

### Advanced Techniques Used

1. **Class Imbalance Handling**
   - Stratified sampling to preserve anomaly ratio
   - Contamination parameter tuning
   - Custom evaluation metrics (no accuracy!)

2. **Feature Engineering**
   - StandardScaler for normalization
   - PCA for visualization (not modeling)
   - Feature importance from Random Forest

3. **Model Evaluation**
   - Precision-Recall curves (preferred over ROC for imbalance)
   - Threshold optimization
   - Cross-validation with F1-scoring

4. **Production Engineering**
   - Input validation and error handling
   - Performance monitoring (latency, throughput)
   - Drift detection (z-score based)
   - Automated retraining triggers

5. **Robustness Testing**
   - Gaussian noise injection (σ ∈ [0, 2])
   - Synthetic anomaly generation (4 types)
   - Learning curve analysis

## 📧 Contact
 
**Email**: mohamadalihousseini@gmail.com  
**LinkedIn**: [my linkedin](https://www.linkedin.com/in/mohamad-ali-husseini-51039a248/)  
**GitHub**: [@Mohamadali0602](https://github.com/Mohamadali0602)

**Institution**: Université Côte d'Azur  
**Program**: Master's Degree in Applied Mathematics  
**Year**: 2026

---

## 🙏 Acknowledgments

- **Dataset**: Machine Learning Group - ULB
- **Inspiration**: Original Isolation Forest paper by Liu et al. (2008)
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, Flask

<p align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ for advancing anomaly detection research
</p>
