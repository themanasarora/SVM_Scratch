# SVM_Scratch

This repository contains a from-scratch implementation of Support Vector Machines (SVM) in Python.  
The goal is to demonstrate the inner workings of SVM — both linear and kernel-based — without relying on external machine-learning libraries like scikit-learn.

---

## 🧠 Overview

Support Vector Machines are powerful supervised learning models used for classification and regression tasks.  
This project implements SVM using:

✅ Gradient-based learning for the linear SVM  
✅ Kernel trick support (e.g., RBF kernel)  
✅ Custom prediction and visualization  
✅ Simple dataset experiments  

The implementation demonstrates how SVMs work under the hood, making it ideal for learning and experimentation.

---

## 📦 Repository Contents

| File | Description |
|------|-------------|
| `SVM.py` | Linear SVM implementation (hinge loss, gradient updates) |
| `SVMkernel.py` | Kernel SVM (dual form, RBF / linear kernels) |
| `svmtest.py` | Example usage with plotting on toy datasets |
| `kernel.py` | Kernel SVM test with nonlinear data |
| `__pycache__/` | Compiled Python cache files |

---

## 🛠️ Features

### Linear SVM
- Manual weight and bias updates
- Hinge loss implementation from scratch
- Visualizes decision boundary and margins

### Kernel SVM
- Dual form with alpha coefficients
- RBF (Gaussian) and linear kernels
- Meshgrid visualization for nonlinear decision regions

---

## 🧪 How To Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/themanasarora/SVM_Scratch.git
