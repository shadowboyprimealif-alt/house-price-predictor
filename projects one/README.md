# 🏠 House Price Prediction - Advanced Scikit-learn Pipeline

This repository contains a comprehensive Machine Learning project designed to predict house prices based on historical housing data from King County, USA. The core focus of this project is to demonstrate a production-grade approach using **Scikit-learn Pipelines** and **ColumnTransformers** to ensure scalability and reproducibility.

---

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Data Analysis & Features](#data-analysis--features)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Tech Stack](#tech-stack)
5. [How to Use](#how-to-use)
6. [Model Performance](#model-performance)

---

## 📌 Introduction
Predicting house prices is a complex regression task influenced by various factors like location, size, and condition. This project moves away from messy, manual data processing and adopts a **Pipeline-based approach**. This ensures that every transformation applied to the training data is automatically and consistently applied to any new input data.

---

## 📊 Data Analysis & Features
The dataset consists of multiple feature types that require different preprocessing strategies:

### 1. Numerical Features:
Features like `sqft_living`, `bedrooms`, `bathrooms`, and `yr_built`.
- **Handling Outliers/Missing Values**: Using `SimpleImputer` with the median strategy.
- **Normalization**: Applying `StandardScaler` to ensure all features are on the same scale for the model.

### 2. Categorical Features:
Features like `waterfront`, `view`, and `zipcode`.
- **Encoding**: Using `OneHotEncoder` to convert categorical data into a format understandable by the model. 
- **Unknown Handling**: Configured to ignore unknown categories in new data to prevent crashes.

---

## 🏗 Pipeline Architecture
The project follows a modular design:

1. **Pre-processing Layer**:
   - `ColumnTransformer` splits the data into numeric and categorical paths.
   - It performs imputation, scaling, and encoding simultaneously.

2. **Modeling Layer**:
   - Uses `RandomForestRegressor`, a powerful ensemble method that handles non-linear relationships and interactions between features effectively.

---

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Libraries**:
  - `Scikit-learn`: For the entire ML workflow.
  - `Pandas`: For data manipulation and CSV handling.
  - `NumPy`: For numerical operations.
  - `Joblib`: For model serialization (saving/loading).

---

## 🚀 How to Use

### Installation
```bash
pip install pandas scikit-learn numpy joblib