# AI/ML Internship Tasks - DevelopersHub Corporation

This repository contains solutions to 3 AI/ML internship tasks completed as part of the DevelopersHub Internship Program. Each task demonstrates core machine learning skills including data analysis, model training, visualization, and API integration.

---

## ✅ Task 1: Exploring and Visualizing the Iris Dataset

**Objective:**  
Load, inspect, and visualize the Iris dataset to understand data trends and feature distributions.

**Dataset Used:**  
- Iris Dataset (loaded via Seaborn or CSV)

**Key Steps:**  
- Loaded dataset using `pandas`  
- Explored structure using `.head()`, `.info()`, `.describe()`  
- Visualized relationships using:
  - Scatter plots (feature correlation)
  - Histograms (value distributions)
  - Box plots (outlier detection)  

**Tools Used:**  
`pandas`, `matplotlib`, `seaborn`

**Key Findings:**  
- Petal length and width are strongly correlated and useful for classification  
- Several outliers observed in sepal width  
- Feature clusters visible per Iris species

---

## ✅ Task 2: Predicting Future Stock Prices (Short-Term)

**Objective:**  
Use historical stock data to predict the next day's closing price.

**Dataset Used:**  
- Stock data for Tesla (`TSLA`) from Yahoo Finance via `yfinance` API

**Models Applied:**  
- Linear Regression  
- Random Forest Regressor

**Key Steps:**  
- Used `Open`, `High`, `Low`, `Volume` to predict next day’s `Close`  
- Data split into train/test using `train_test_split`  
- Plotted actual vs predicted prices

**Tools Used:**  
`yfinance`, `scikit-learn`, `pandas`, `matplotlib`

**Key Results:**  
- Random Forest performed better than Linear Regression  
- Visual plots showed close alignment for short-term predictions

---

## ✅ Task 3: Heart Disease Prediction

**Objective:**  
Predict if a person is at risk of heart disease using health indicators.

**Dataset Used:**  
- UCI Heart Disease Dataset (`processed.cleveland.data`)

**Models Applied:**  
- Logistic Regression  
- Decision Tree Classifier

**Key Steps:**  
- Cleaned missing values (`?` → NaN → numeric)  
- Performed EDA (heatmap, histograms, class distribution)  
- Binary classification (`0 = no disease`, `1 = disease`)  
- Model evaluation using accuracy, ROC-AUC, confusion matrix  
- Feature importance analysis for Decision Tree

**Tools Used:**  
`pandas`, `matplotlib`, `seaborn`, `scikit-learn`

**Key Results:**  
- Logistic Regression Accuracy: ~85%  
- Decision Tree ROC-AUC: ~0.91  
- Top features: `thalach`, `oldpeak`, `ca`, and `cp`

---

## 📁 Folder Structure

