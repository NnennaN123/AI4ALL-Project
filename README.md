# AI4ALL-Project

Link to our repo: https://github.com/NnennaN123/AI4ALL-Project

Link to Streamlit App: https://cranberryai4allproject.streamlit.app/

CRANBerry Team's Project!

## Overview

This project builds machine learning models to predict whether a location (from NREL's Wind Toolkit) has a wind turbine based on various features. The workflow includes data loading, spatial matching, feature engineering, model training, and evaluation.

## Project Timeline & Model Evolution

This project demonstrates an iterative machine learning workflow, progressively improving model performance through algorithm selection, feature engineering, and hyperparameter optimization.

### Phase 1: Baseline Models (Initial Exploration)

#### 1. Logistic Regression Model

**Location:** `logistic_regression/logistic_regression.ipynb`

The initial baseline model uses **Logistic Regression** to establish a performance benchmark. This linear model was chosen for its interpretability and computational efficiency.

**Model Configuration:**
- **Algorithm:** Logistic Regression with balanced class weights
- **Features:** 4 base features
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

**Key Finding:** Good recall for detecting turbines (72.8%) but lower precision (52.2%), indicating many false positives. This established the baseline for comparison.

---

#### 2. Random Forest Model

**Location:** `random_forest/random_forest.ipynb`

The second iteration uses **Random Forest**, an ensemble method that combines multiple decision trees to capture non-linear relationships and improve prediction accuracy.

**Model Configuration:**
- **Algorithm:** Random Forest Classifier
- **Features:** Same 4 base features as Logistic Regression
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

**Improvement over Logistic Regression:**
- **+5.2% improvement in ROC-AUC** (0.770 vs 0.732)
- **+6.0% improvement in accuracy** (0.703 vs 0.643)
- Better overall F1-scores for both classes

**Key Finding:** Better overall performance but more conservative in predicting turbines (44.8% recall vs 72.8%), suggesting the need for more sophisticated algorithms.

---

#### 3. XGBoost Model (Initial)

**Location:** `xgboost/xgboost.ipynb`

The third iteration uses **XGBoost** (Extreme Gradient Boosting), a powerful gradient boosting framework known for superior performance in structured data problems.

**Model Configuration:**
- **Algorithm:** XGBoost Classifier
- **Features:** Same 4 base features as previous models
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

**Improvement over Previous Models:**
- **+10.0% improvement in ROC-AUC** over Random Forest (0.847 vs 0.770)
- **+15.8% improvement in ROC-AUC** over Logistic Regression (0.847 vs 0.732)
- **+6.3% improvement in accuracy** over Random Forest (0.766 vs 0.703)
- Best overall balance between precision and recall

**Key Finding:** XGBoost showed the best performance, confirming it as the optimal algorithm. However, further improvements were possible through feature engineering.

---

### Phase 2: Feature Engineering & Optimization

#### 4. Feature Engineering Exploration

**Location:** `feature_engineering.ipynb`

Conducted systematic feature engineering to identify the optimal feature combination. Tested **12 different feature configurations** (X_train_1 through X_train_12) including:

- **Base features:** `fraction_of_usable_area`, `capacity_factor`
- **Numeric features:** `wind_speed`, `capacity`
- **Categorical features:** `wind_speed_category`, `capacity_category` (converted to numeric)
- **Geographic features:** `State` (one-hot encoded, ~50+ columns) or `Region` (one-hot encoded, ~7 columns)
- **Engineered interaction features:**
  - `combined_wind_rescource` = wind_speed × capacity_factor
  - `potential_with_constraints` = capacity × fraction_of_usable_area

**Key Findings:**
- **Best feature combination: X_train_5** (Base + Numeric + State + New Features)
  - ROC-AUC: 0.9305 (before hyperparameter tuning)
  - Accuracy: 0.8548
  - F1-Score: 0.8134
- Geographic features (State) provided significant performance boost (+0.05-0.08 ROC-AUC)
- Numeric features outperformed categorical versions
- Interaction features added value, especially when combined with geographic features

**Documentation:** See `X_TRAIN_COMBINATIONS_README.md` for complete analysis of all 12 feature combinations.

---

#### 5. Final Model: Feature-Engineered XGBoost with Hyperparameter Tuning 

**Location:** `xgboost/xgboost_hyperparameter_tuning.ipynb`

**Model File:** `xgboost/xgboost_tuned_feat_eng_wind_model.pkl`

The final production model combines the best feature engineering (X_train_5) with systematic hyperparameter optimization using GridSearchCV.

**Model Configuration:**
- **Algorithm:** XGBoost Classifier (tuned)
- **Features:** 56 features (X_train_5 configuration)
  - Base: `fraction_of_usable_area`, `capacity_factor`
  - Numeric: `wind_speed`, `capacity`
  - Geographic: `State` (one-hot encoded, ~50+ columns)
  - Engineered: `combined_wind_rescource`, `potential_with_constraints`
- **Preprocessing:** No scaling required (tree-based models are scale-invariant)
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
  - Tested 54 parameter combinations
  - Scoring metric: ROC-AUC
- **Best Hyperparameters:**
  - `max_depth`: 8
  - `learning_rate`: 0.1
  - `n_estimators`: 300
  - `subsample`: 0.7
  - `colsample_bytree`: 1.0
  - `scale_pos_weight`: 1.6201 (handles class imbalance)

**Final Performance Metrics:**
- **ROC-AUC Score:** **0.9545** ⭐
- **Accuracy:** **0.8783** ⭐
- **F1-Score:** **0.8537** ⭐
- **Precision (No Turbine):** 0.960 | **Recall:** 0.840 | **F1-Score:** 0.896
- **Precision (Turbine):** 0.780 | **Recall:** 0.943 | **F1-Score:** 0.854

**Improvement Journey:**
- **+12.7% improvement in ROC-AUC** over initial XGBoost (0.9545 vs 0.847)
- **+11.2% improvement in accuracy** over initial XGBoost (0.8783 vs 0.766)
- **+30.4% improvement in ROC-AUC** over Logistic Regression baseline (0.9545 vs 0.732)
- **+23.5% improvement in accuracy** over Logistic Regression baseline (0.8783 vs 0.643)

**Key Achievements:**
- Excellent turbine recall (94.3%) - captures nearly all turbine locations
- High precision for both classes (96.0% for No Turbine, 78.0% for Turbine)
- Best-in-class performance across all metrics
- Production-ready model deployed in Streamlit app

---

## Complete Model Comparison

| Metric | Logistic Regression | Random Forest | XGBoost (Initial) | **XGBoost (Final)** | Winner |
|--------|---------------------|---------------|-------------------|---------------------|--------|
| **ROC-AUC** | 0.732 | 0.770 | 0.847 | **0.9545** | **Final XGBoost** |
| **Accuracy** | 0.643 | 0.703 | 0.766 | **0.8783** | **Final XGBoost** |
| **F1-Score** | 0.640 | 0.658 | 0.746 | **0.8537** | **Final XGBoost** |
| **Turbine Recall** | 0.728 | 0.448 | 0.645 | **0.943** | **Final XGBoost** |
| **Turbine Precision** | 0.522 | 0.663 | 0.708 | **0.780** | **Final XGBoost** |
| **No Turbine F1** | 0.672 | 0.782 | 0.817 | **0.896** | **Final XGBoost** |

**Performance Evolution:**
- **Phase 1 (Baseline):** Logistic Regression → Random Forest → XGBoost
  - ROC-AUC: 0.732 → 0.770 → 0.847 (+15.7% total improvement)
- **Phase 2 (Optimization):** Feature Engineering → Hyperparameter Tuning
  - ROC-AUC: 0.847 → 0.9305 → **0.9545** (+12.7% additional improvement)
- **Total Improvement:** +30.4% ROC-AUC from baseline to final model

**Key Insights:**
- The **final feature-engineered XGBoost model** achieves the best performance across all metrics
- Feature engineering provided the largest performance boost (+0.0835 ROC-AUC)
- Hyperparameter tuning further refined the model (+0.0240 ROC-AUC)
- The iterative approach demonstrates systematic ML engineering practices
- Final model achieves excellent balance: 94.3% recall for turbines with 78.0% precision

---

## Technical Highlights

### Feature Engineering Process
- **Systematic testing:** 12 feature combinations evaluated
- **Geographic features:** State-level encoding proved most valuable
- **Interaction features:** Captured non-linear relationships between variables
- **Feature selection:** Numeric features outperformed categorical discretizations

### Hyperparameter Optimization
- **Method:** GridSearchCV with 5-fold cross-validation
- **Search space:** 54 parameter combinations tested
- **Optimization metric:** ROC-AUC (appropriate for imbalanced classification)
- **Result:** Significant improvement over default parameters

### Model Selection Strategy
1. **Algorithm comparison:** Logistic Regression → Random Forest → XGBoost
2. **Feature engineering:** Systematic evaluation of 12 configurations
3. **Hyperparameter tuning:** Grid search on best feature set
4. **Final validation:** Test set performance confirms production readiness

---

## Data

The project uses:
- **USWTDB** (US Wind Turbine Database): Contains information about existing wind turbines
- **NREL WTK** (Wind Toolkit): Grid cell locations and wind resource data

Spatial matching is performed using geospatial joins to match turbines to NREL grid cells within a 25 km radius.

---

## Project Structure

- `logistic_regression/`: Baseline Logistic Regression model
- `random_forest/`: Random Forest model for comparison
- `xgboost/`: 
  - `xgboost.ipynb`: Initial XGBoost model (4 features)
  - `xgboost_hyperparameter_tuning.ipynb`: Final optimized model with feature engineering
  - `xgboost_tuned_feat_eng_wind_model.pkl`: **Production model**
  - `xgboost_tuned_feat_eng_model_metrics.json`: Final model metrics
- `feature_engineering.ipynb`: Systematic feature engineering exploration
- `X_TRAIN_COMBINATIONS_README.md`: Complete documentation of 12 feature combinations tested
- `datasets/`: Training and source data files
- `streamlit/`: Production Streamlit application

---

## Deployment

The final model is deployed in a Streamlit web application:
- **Live App:** https://cranberryai4allproject.streamlit.app/
- **Model:** Feature-engineered XGBoost with hyperparameter tuning
- **Performance:** ROC-AUC 0.9545, Accuracy 0.8783
