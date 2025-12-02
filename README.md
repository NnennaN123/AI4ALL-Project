# AI4ALL-Project

Link to our repo: https://github.com/NnennaN123/AI4ALL-Project

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

### 3. XGBoost Model

**Location:** `xgboost/xgboost.ipynb`

The third model uses **XGBoost** (Extreme Gradient Boosting), a powerful gradient boosting framework that combines multiple weak learners (decision trees) sequentially to create a strong predictive model. XGBoost is known for its superior performance and efficiency.

**Model Configuration:**
- **Algorithm:** XGBoost Classifier
- **Features:** Same as previous models
  - `fraction_of_usable_area`
  - `capacity`
  - `wind_speed`
  - `capacity_factor`
- **Preprocessing:** No scaling required (tree-based models are scale-invariant)
- **Hyperparameters:**
  - `n_estimators=300` (number of boosting rounds)
  - `learning_rate=0.05` (step size shrinkage)
  - `max_depth=6` (maximum tree depth)
  - `subsample=0.8` (row subsampling ratio)
  - `colsample_bytree=0.8` (column subsampling ratio)
  - `objective='binary:logistic'` (binary classification)
  - `eval_metric='logloss'` (evaluation metric)

**Performance Metrics:**
- **ROC-AUC Score:** 0.847
- **Accuracy:** 0.766
- **Precision (No Turbine):** 0.796 | **Recall:** 0.839 | **F1-Score:** 0.817
- **Precision (Turbine):** 0.708 | **Recall:** 0.645 | **F1-Score:** 0.675

The XGBoost model achieves the best performance across all models with:
- **+10.0% improvement in ROC-AUC** over Random Forest (0.847 vs 0.770)
- **+15.8% improvement in ROC-AUC** over Logistic Regression (0.847 vs 0.732)
- **+6.3% improvement in accuracy** over Random Forest (0.766 vs 0.703)
- Best overall balance between precision and recall for both classes

The model demonstrates strong performance in detecting turbines while maintaining good precision, making it the best choice for overall predictive performance.

---

## Model Comparison

| Metric | Logistic Regression | Random Forest | XGBoost | Winner |
|--------|---------------------|---------------|---------|--------|
| **ROC-AUC** | 0.732 | 0.770 | 0.847 | XGBoost |
| **Accuracy** | 0.643 | 0.703 | 0.766 | XGBoost |
| **Turbine Recall** | 0.728 | 0.448 | 0.645 | Logistic Regression |
| **Turbine Precision** | 0.522 | 0.663 | 0.708 | XGBoost |
| **No Turbine F1** | 0.672 | 0.782 | 0.817 | XGBoost |

**Key Insights:**
- The **XGBoost model** achieves the best overall performance with the highest ROC-AUC (0.847) and accuracy (0.766)
- **Logistic Regression** has the highest recall for turbines (0.728), making it best at finding all potential turbine locations (fewer false negatives)
- **XGBoost** provides the best balance between precision and recall, with the highest turbine precision (0.708) and strong recall (0.645)
- **Random Forest** shows good overall performance but is more conservative in predicting turbines
- The choice between models depends on the use case:
  - **XGBoost**: Best overall performance and balanced metrics (recommended)
  - **Logistic Regression**: Prioritize finding all turbines (higher recall)
  - **Random Forest**: Good balance between performance and interpretability

## Data

The project uses:
- **USWTDB** (US Wind Turbine Database): Contains information about existing wind turbines
- **NREL WTK** (Wind Toolkit): Grid cell locations and wind resource data

Spatial matching is performed using geospatial joins to match turbines to NREL grid cells within a 25 km radius.

## Files

- `logistic_regression/`: Contains the Logistic Regression model notebook, trained model, scaler, and metrics
- `random_forest/`: Contains the Random Forest model notebook, trained model, and metrics
- `xgboost/`: Contains the XGBoost model notebook, trained model, and metrics
- `datasets/`: Contains the training and source data files