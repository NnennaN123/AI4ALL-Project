# Feature Importance Analysis for Wind Turbine Prediction Model

This document analyzes which features are important for building the wind turbine prediction model, based on current model usage, feature importance analysis, and domain knowledge.

---

## Currently Used Features (All Models)

All three models (Logistic Regression, Random Forest, and XGBoost) currently use **4 core features**:

### 1. **fraction_of_usable_area** ⭐⭐⭐
- **Definition**: Fraction of grid cell usable for wind development (0-1)
- **Importance**: Supporting role (~15-20% importance in Random Forest)
- **Why Important**: 
  - Accounts for land use restrictions, terrain constraints, and regulatory limitations
  - Higher usable area = more potential for turbine installation
  - Captures practical feasibility beyond just wind resource quality

### 2. **capacity** ⭐⭐⭐
- **Definition**: Potential wind power capacity of the site (MW)
- **Importance**: Supporting role (~15-20% importance in Random Forest)
- **Why Important**:
  - Indicates the maximum power output potential
  - Higher capacity sites are more attractive for development
  - Correlates with economic viability

### 3. **wind_speed** ⭐⭐⭐⭐⭐
- **Definition**: Average wind speed at the site (m/s)
- **Importance**: **Second most important** (~31% importance in Random Forest)
- **Why Important**:
  - Direct indicator of wind resource quality
  - Higher wind speeds = more energy generation potential
  - Primary driver of economic feasibility
  - Strongest predictor after capacity factor

### 4. **capacity_factor** ⭐⭐⭐⭐⭐
- **Definition**: Expected capacity factor (0-1), ratio of actual to maximum possible output
- **Importance**: **Most important feature** (~38% importance in Random Forest)
- **Why Important**:
  - Represents overall efficiency and economic viability
  - Combines wind speed, turbine technology, and site characteristics
  - Best single predictor of turbine presence
  - Higher capacity factor = better return on investment

---

## Feature Importance Ranking (Based on Random Forest Analysis)

According to the Random Forest model's feature importance analysis:

1. **Capacity Factor** - 38% (Most Important)
2. **Wind Speed** - 31% (Second Most Important)
3. **Capacity** - ~15-20% (Supporting)
4. **Fraction of Usable Area** - ~15-20% (Supporting)

**Key Insight**: Capacity Factor and Wind Speed together account for ~69% of the model's predictive power, making them the critical features for turbine location prediction.

---

## Features NOT Currently Used (But Potentially Valuable)

### Geographic Features

#### **State** ⭐⭐
- **Potential Value**: Could capture regional policy differences, regulatory environments, and historical development patterns
- **Limitation**: High cardinality (50+ categories), would require encoding
- **Recommendation**: Could be useful as a categorical feature with one-hot encoding or target encoding

#### **County** ⭐
- **Potential Value**: More granular than State, could capture local zoning and land use patterns
- **Limitation**: Very high cardinality (3000+ counties), likely to cause overfitting
- **Recommendation**: Probably too granular; State is more practical

#### **longitude / latitude** ⭐⭐
- **Potential Value**: Could capture geographic patterns, climate zones, and regional wind patterns
- **Limitation**: 
  - Models might learn geographic clustering rather than causal relationships
  - Could introduce geographic bias
- **Recommendation**: Could be useful for capturing regional effects, but may need careful handling to avoid overfitting to specific regions

### Temporal Features (from USWTDB)

#### **p_year** (Project Year) ⭐⭐⭐
- **Potential Value**: Could capture temporal trends in turbine development
- **Limitation**: Only available for existing turbines, not for prediction on new sites
- **Recommendation**: Useful for analysis but not for prediction on new locations

#### **t_retro_yr** (Retrofit Year) ⭐
- **Potential Value**: Indicates turbine upgrades
- **Limitation**: Only available for existing turbines, sparse data
- **Recommendation**: Not useful for prediction

### Turbine Characteristics (from USWTDB)

**Note**: These features (t_hh, t_rd, t_cap, t_manu, t_model, etc.) are only available for **existing turbines**, not for new site predictions. They cannot be used to predict whether a location will have a turbine because they describe turbines that already exist.

However, they could be useful for:
- **Analysis**: Understanding characteristics of existing turbines
- **Post-prediction**: If predicting turbine specifications after determining a site is suitable

### Project Information (from USWTDB)

#### **p_tnum** (Number of Turbines in Project) ⭐⭐
- **Potential Value**: Could indicate project scale preferences
- **Limitation**: Only available for existing projects
- **Recommendation**: Not directly useful for prediction, but could inform analysis

#### **p_cap** (Project Capacity) ⭐⭐
- **Potential Value**: Similar to individual capacity
- **Limitation**: Only available for existing projects
- **Recommendation**: Not directly useful for prediction

### Data Quality Features

#### **t_conf_atr** (Turbine Confidence Attribute) ⭐
- **Potential Value**: Could filter low-confidence records
- **Recommendation**: Useful for data quality control, not for prediction

#### **t_conf_loc** (Location Confidence) ⭐
- **Potential Value**: Could filter low-confidence locations
- **Recommendation**: Useful for data quality control, not for prediction

### Matching Features (from Joined Dataset)

#### **dist_m** (Distance in meters) ⭐⭐
- **Potential Value**: Distance between turbine and matched grid cell
- **Limitation**: Only available after spatial matching
- **Recommendation**: Could be useful as a feature to indicate match quality, but might introduce data leakage if not handled carefully

---

## Recommendations for Feature Engineering

### High Priority Additions

1. **Geographic Region Encoding** ⭐⭐⭐
   - Create regional categories (e.g., "Great Plains", "West Coast", "Northeast")
   - Could capture regional wind patterns and policy differences
   - More interpretable than raw coordinates

2. **Wind Speed Categories** ⭐⭐⭐
   - Create bins for wind speed (e.g., "Low", "Medium", "High", "Very High")
   - Could help models capture non-linear relationships
   - More interpretable than continuous values

3. **Capacity Factor Categories** ⭐⭐
   - Similar to wind speed, create categorical bins
   - Could help with interpretability

### Medium Priority Additions

4. **Interaction Features** ⭐⭐⭐
   - `wind_speed × capacity_factor`: Combined wind resource quality
   - `capacity × fraction_of_usable_area`: Total potential considering constraints
   - Could capture non-linear relationships

5. **Normalized Features** ⭐⭐
   - Capacity per unit area
   - Wind speed normalized by region average

### Low Priority / Experimental

6. **State as Categorical** ⭐⭐
   - One-hot encoding or target encoding
   - Could capture policy and regulatory differences
   - Risk of overfitting

7. **Distance to Nearest Turbine** ⭐⭐
   - Could indicate clustering patterns
   - Requires careful feature engineering to avoid data leakage

---

## Features to Avoid

### ❌ **Not Useful for Prediction**:
- **USWTDB turbine characteristics** (t_hh, t_rd, t_cap, t_manu, t_model): Only available for existing turbines
- **Project information** (p_name, p_year, p_tnum): Only available for existing projects
- **ID fields** (case_id, site_id, faa_ors, etc.): Not predictive, only identifiers
- **File paths** (full_timeseries_directory, full_timeseries_path): Not predictive
- **power_curve**: Already dropped in current models (likely redundant with capacity_factor)

### ⚠️ **Use with Caution**:
- **Raw coordinates** (longitude, latitude): Risk of geographic overfitting
- **County**: Too granular, high cardinality
- **dist_m**: Could introduce data leakage if not handled properly

---

## Summary: Essential Features for Model Building

### Core Features (Currently Used - Keep These) ✅

1. **capacity_factor** - Most important (38%)
2. **wind_speed** - Second most important (31%)
3. **capacity** - Supporting feature (~15-20%)
4. **fraction_of_usable_area** - Supporting feature (~15-20%)

### Potential Additions (Consider for Improvement) 🔄

1. **Geographic region** (derived from State or coordinates)
2. **Wind speed categories** (binned from wind_speed)
3. **Interaction features** (wind_speed × capacity_factor, etc.)

### Features to Exclude ❌

- All USWTDB turbine-specific features (only available post-installation)
- ID fields and file paths
- power_curve (redundant)

---

## Model Performance Context

Current model performance with 4 features:
- **XGBoost**: ROC-AUC 0.847, Accuracy 0.766 (Best)
- **Random Forest**: ROC-AUC 0.770, Accuracy 0.703
- **Logistic Regression**: ROC-AUC 0.732, Accuracy 0.643

The 4 current features provide strong predictive power. Additional features may provide marginal improvements but should be carefully evaluated to avoid overfitting, especially geographic features that might learn location-specific patterns rather than generalizable relationships.

---

## Next Steps for Feature Engineering

1. **Test geographic region encoding** - Create regional categories and test impact
2. **Create interaction features** - Test wind_speed × capacity_factor
3. **Feature selection analysis** - Use recursive feature elimination to validate current feature set
4. **Cross-validation** - Ensure new features improve generalization, not just training performance
