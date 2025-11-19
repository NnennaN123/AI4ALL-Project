# AI4ALL-Project
CRANBerry Team's Project!

## Overview

This project builds machine learning models to predict whether a location (from NREL's Wind Toolkit) has a wind turbine based on various features. The workflow includes data loading, spatial matching, feature engineering, model training, and evaluation.

## Models

### 1. Logistic Regression Model

**Location:** `logistic_regression/logistic_regression.ipynb`

The initial model uses **Logistic Regression** to predict wind turbine presence. This linear model was chosen as a baseline due to its interpretability and efficiency.

**Model Configuration:**
- **Algorithm:** Logistic Regression with balanced class weights
- **Features:** 
  - `fraction_of_usable_area`: Fraction of grid cell usable for wind development
  - `capacity`: Potential capacity of the site
  - `wind_speed`: Average wind speed at the site
  - `capacity_factor`: Expected capacity factor (efficiency)
- **Preprocessing:** StandardScaler for feature normalization
- **Hyperparameters:**
  - `max_iter=1000`
  - `class_weight="balanced"` (to handle class imbalance)
  - `n_jobs=-1` (parallel processing)

**Performance Metrics:**
- **ROC-AUC Score:** 0.732
- **Accuracy:** 0.643
- **Precision (No Turbine):** 0.779 | **Recall:** 0.591 | **F1-Score:** 0.672
- **Precision (Turbine):** 0.522 | **Recall:** 0.728 | **F1-Score:** 0.608

The model shows good recall for detecting turbines (72.8%) but lower precision (52.2%), indicating it tends to predict more false positives. The balanced class weights help address the class imbalance in the dataset.

---

### 2. Random Forest Model

**Location:** `random_forest/random_forest.ipynb`

The second model uses **Random Forest**, an ensemble method that combines multiple decision trees to improve prediction accuracy and handle non-linear relationships.

**Model Configuration:**
- **Algorithm:** Random Forest Classifier
- **Features:** Same as Logistic Regression model
  - `fraction_of_usable_area`
  - `capacity`
  - `wind_speed`
  - `capacity_factor`
- **Preprocessing:** No scaling required (tree-based models are scale-invariant)
- **Hyperparameters:**
  - `n_estimators=500` (number of trees)
  - `max_leaf_nodes=16` (limits tree depth)
  - `n_jobs=-1` (parallel processing)
  - `random_state=42` (reproducibility)

**Performance Metrics:**
- **ROC-AUC Score:** 0.770
- **Accuracy:** 0.703
- **Precision (No Turbine):** 0.717 | **Recall:** 0.860 | **F1-Score:** 0.782
- **Precision (Turbine):** 0.663 | **Recall:** 0.448 | **F1-Score:** 0.535

The Random Forest model outperforms the Logistic Regression model with:
- **+5.2% improvement in ROC-AUC** (0.770 vs 0.732)
- **+6.0% improvement in accuracy** (0.703 vs 0.643)
- Better overall F1-scores for both classes

However, it shows lower recall for detecting turbines (44.8% vs 72.8%), meaning it's more conservative in predicting turbine presence but more precise when it does.

---

## Model Comparison

| Metric | Logistic Regression | Random Forest | Winner |
|--------|---------------------|---------------|--------|
| **ROC-AUC** | 0.732 | 0.770 | Random Forest |
| **Accuracy** | 0.643 | 0.703 | Random Forest |
| **Turbine Recall** | 0.728 | 0.448 | Logistic Regression |
| **Turbine Precision** | 0.522 | 0.663 | Random Forest |
| **No Turbine F1** | 0.672 | 0.782 | Random Forest |

**Key Insights:**
- The Random Forest model achieves better overall performance with higher ROC-AUC and accuracy
- Logistic Regression has higher recall for turbines, making it better at finding all potential turbine locations (fewer false negatives)
- Random Forest has higher precision for turbines, making it more reliable when it predicts a turbine exists (fewer false positives)
- The choice between models depends on the use case: prioritize finding all turbines (Logistic Regression) vs. minimizing false positives (Random Forest)

## Data

The project uses:
- **USWTDB** (US Wind Turbine Database): Contains information about existing wind turbines
- **NREL WTK** (Wind Toolkit): Grid cell locations and wind resource data

Spatial matching is performed using geospatial joins to match turbines to NREL grid cells within a 25 km radius.

## Files

- `logistic_regression/`: Contains the Logistic Regression model notebook, trained model, scaler, and metrics
- `random_forest/`: Contains the Random Forest model notebook, trained model, and metrics
- `datasets/`: Contains the training and source data files