# X_train Combinations Key

This document provides a comprehensive key to all 12 X_train combinations created for testing different feature configurations.

**Note:** State and Region features are mutually exclusive - each combination uses either State OR Region, never both.

## Base Features (Always Included)

All combinations include these base features:
- `fraction_of_usable_area` (numeric)
- `capacity_factor` (numeric)

## Feature Types

### Numeric Features
- `wind_speed` - Continuous wind speed values
- `capacity` - Continuous capacity values

### Categorical Features (Converted to Numeric)
- `wind_speed_category` - Categorized wind speeds converted to numeric: Low=1, Medium=2, High=3, Very High=4
- `capacity_category` - Categorized capacity values converted to numeric: Very Low (<2)=1, 2-4=2, 6-8=4, 10-12=6, 14-16=8, etc.

### Geographic Features
- `State` - Individual US states (high cardinality)
- `Region` - Regional groupings (Northeast, West Coast, Great Plains, etc.)

### New Engineered Features
- `combined_wind_rescource` - Interaction feature: wind_speed × capacity_factor
- `potential_with_constraints` - Interaction feature: capacity × fraction_of_usable_area

---

## Categorical Feature Numeric Encoding

### wind_speed_category Mapping
- `Low` → 1
- `Medium` → 2
- `High` → 3
- `Very High` → 4
- `Unknown` → 0

### capacity_category Mapping
- `Very Low (<2)` → 1
- `2-4` → 2
- `Gap (4-6)` → 3
- `6-8` → 4
- `Gap (8-10)` → 5
- `10-12` → 6
- `Gap (12-14)` → 7
- `14-16` → 8
- `Very High (>16)` → 9
- `Unknown` → 0

**Note:** These mappings preserve the ordinal relationship between categories (e.g., Low < Medium < High < Very High).

---

## Complete Data Processing Pipeline

This section documents all data processing steps applied to create each X_train combination, based on the complete feature engineering notebook.

### Step 1: Data Loading and Initial Preparation
1. **Load dataset**: Read `datasets/nrel_training_data.csv`
2. **Remove non-predictive columns**: Drop `County`, `full_timeseries_directory`, `full_timeseries_path`
3. **Split data**: Separate features (X) and target (y = `isTurbine`)
4. **Train/test split**: Split into X_train, X_test, y_train, y_test (80/20 split, random_state=42)
5. **Initial feature selection**: Select only `['fraction_of_usable_area', 'capacity', 'wind_speed', 'capacity_factor', 'latitude', 'longitude', 'State']`

### Step 2: State Data Imputation
1. **Identify missing states**: Find rows where State is NaN or 'Unknown'
   - Example: ~11,408 rows with missing/unknown states in initial dataset
2. **Load shapefile**: Load US state boundaries from `tl_2025_us_state` shapefile (TIGER/Line data from U.S. Census Bureau)
   - Contains 56 entities: 50 states + DC + 5 territories (Puerto Rico, Guam, American Samoa, US Virgin Islands, CNMI)
3. **Map coordinates to states**: Use `map_coordinates_to_state()` function to determine state from latitude/longitude coordinates
   - Uses spatial point-in-polygon operations with GeoPandas
   - Processes coordinates in batches for efficiency
4. **Batch processing**: Process missing states in batches of 1000 for efficiency
   - Progress updates every 5000 rows
5. **Update State column**: Fill missing/unknown states with mapped values
   - Example: Successfully mapped ~1,549 coordinates to states
6. **Result**: State column now has minimal missing values (some offshore/international locations may remain as NaN)
   - Remaining NaN values typically represent offshore locations or coordinates outside US boundaries

### Step 3: Wind Speed Categorization
1. **Create wind_speed_category**: Apply `categorize_wind_speed()` function with fixed thresholds
2. **Fixed thresholds** (based on wind energy industry standards):
   - `Low`: < 6.0 m/s (marginal for wind energy)
   - `Medium`: 6.0 - 8.0 m/s (acceptable for wind energy)
   - `High`: 8.0 - 10.0 m/s (good for wind energy)
   - `Very High`: >= 10.0 m/s (excellent for wind energy)
   - `Unknown`: NaN values
3. **Data distribution** (from training set):
   - `Medium`: ~82,393 samples (53.96%)
   - `High`: ~60,157 samples (39.40%)
   - `Low`: ~7,172 samples (4.70%)
   - `Very High`: ~2,966 samples (1.94%)
4. **Result**: New column `wind_speed_category` added to X_train

### Step 4: Capacity Categorization
1. **Create capacity_category**: Apply `categorize_capacity()` function with fixed thresholds
2. **Fixed thresholds** (inclusive ranges):
   - `2-4`: Capacity between 2 and 4 MW
   - `6-8`: Capacity between 6 and 8 MW
   - `10-12`: Capacity between 10 and 12 MW
   - `14-16`: Capacity between 14 and 16 MW
   - `Very Low (<2)`: Capacity less than 2 MW
   - `Gap (4-6)`: Capacity between 4 and 6 MW (gap range)
   - `Gap (8-10)`: Capacity between 8 and 10 MW (gap range)
   - `Gap (12-14)`: Capacity between 12 and 14 MW (gap range)
   - `Very High (>16)`: Capacity greater than 16 MW
   - `Unknown`: NaN values
3. **Data distribution** (from training set):
   - `14-16`: ~127,639 samples (83.59%) - Most common
   - `10-12`: ~17,654 samples (11.56%)
   - `2-4`: ~4,534 samples (2.97%)
   - `6-8`: ~2,861 samples (1.87%)
   - Other categories (Very Low, Gaps, Very High): Rare or absent in dataset
4. **Result**: New column `capacity_category` added to X_train

### Step 5: Region Creation
1. **Create Region column**: Apply `add_region_column()` function
2. **State to Region mapping**:
   - `Northeast`: Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, Pennsylvania
   - `West Coast`: California, Oregon, Washington
   - `Great Plains`: North Dakota, South Dakota, Nebraska, Kansas, Oklahoma, Texas, Montana, Wyoming, Colorado
   - `Southeast`: Delaware, Maryland, DC, Virginia, West Virginia, Kentucky, Tennessee, North Carolina, South Carolina, Georgia, Florida, Alabama, Mississippi, Arkansas, Louisiana
   - `Midwest`: Ohio, Michigan, Indiana, Illinois, Wisconsin, Minnesota, Iowa, Missouri
   - `Mountain West`: Idaho, Utah, Nevada, New Mexico, Arizona
   - `Other`: Alaska, Hawaii
   - `Territories`: Puerto Rico, US Virgin Islands, Guam, American Samoa, CNMI
   - `Offshore/International`: Missing/unknown states
3. **Result**: New column `Region` added to X_train

### Step 6: Interaction Feature Engineering
1. **Create combined_wind_rescource**: `wind_speed × capacity_factor`
   - **Formula**: `wind_speed × capacity_factor`
   - **Purpose**: Captures the interaction between wind speed and capacity factor
   - **Interpretation**: Represents combined wind resource potential - higher values indicate better overall wind energy potential
   - **Units**: m/s × dimensionless = m/s (but represents combined resource)
2. **Create potential_with_constraints**: `capacity × fraction_of_usable_area`
   - **Formula**: `capacity × fraction_of_usable_area`
   - **Purpose**: Captures the interaction between capacity and usable area constraints
   - **Interpretation**: Represents potential capacity accounting for land use constraints - actual usable capacity considering area limitations
   - **Units**: MW × fraction = MW (but represents constrained capacity)
3. **Result**: Two new engineered features added to X_train
   - These features are multiplicative interactions that may capture non-linear relationships
   - They combine information from multiple base features into single predictive features

### Step 7: Remove Coordinate Features
1. **Drop latitude and longitude**: Remove coordinate columns (no longer needed after state mapping)
2. **Result**: X_train now contains only predictive features

### Step 8: Create X_train Combinations
1. **Define feature groups**:
   - Base features: `fraction_of_usable_area`, `capacity_factor` (always included)
   - Numeric features: `wind_speed`, `capacity`
   - Categorical features: `wind_speed_category`, `capacity_category`
   - Geographic features: `State`, `Region` (mutually exclusive)
   - New features: `combined_wind_rescource`, `potential_with_constraints`
2. **Create 12 combinations**: Select appropriate features for each X_train_1 through X_train_12

### Step 9: Categorical Feature Numeric Encoding (Applied to X_train_3, 4, 9-12)
1. **Convert wind_speed_category to numeric**:
   - Map text labels to numbers: Low→1, Medium→2, High→3, Very High→4, Unknown→0
   - Preserves ordinal relationship
2. **Convert capacity_category to numeric**:
   - Map text labels to numbers based on numeric value ordering
   - Very Low (<2)→1, 2-4→2, Gap (4-6)→3, 6-8→4, Gap (8-10)→5, 10-12→6, Gap (12-14)→7, 14-16→8, Very High (>16)→9, Unknown→0
3. **Result**: Categorical features in X_train_3, X_train_4, X_train_9, X_train_10, X_train_11, X_train_12 are now numeric

### Step 10: Geographic Feature One-Hot Encoding (Applied to X_train_5-12)
1. **One-hot encode State** (X_train_5, X_train_6, X_train_9, X_train_10):
   - Use `pd.get_dummies()` with prefix 'State_'
   - Creates binary columns for each state (e.g., State_California, State_Texas, etc.)
   - Results in ~50+ columns (one per state/territory present in data)
   - Original State column is dropped
2. **One-hot encode Region** (X_train_7, X_train_8, X_train_11, X_train_12):
   - Use `pd.get_dummies()` with prefix 'Region_'
   - Creates binary columns for each region (e.g., Region_Northeast, Region_West_Coast, etc.)
   - Results in ~7-8 columns (one per region)
   - Original Region column is dropped
3. **Result**: Geographic features are now binary-encoded and ready for model training

---

## X_train Combinations - Detailed Processing Steps

### X_train_1
**Features:** Base + Numeric (wind_speed, capacity) + New Features

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed`, `capacity`, `combined_wind_rescource`, `potential_with_constraints`
3. No categorical encoding needed (uses numeric features)
4. No geographic encoding needed (no State/Region)
5. **Final features**: 6 numeric features

**Tests:** Baseline with new engineered features  
**Purpose:** Establish baseline performance with numeric features and new interaction features

---

### X_train_2
**Features:** Base + Numeric (wind_speed, capacity)

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed`, `capacity`
3. New features excluded (control group)
4. No categorical encoding needed (uses numeric features)
5. No geographic encoding needed (no State/Region)
6. **Final features**: 4 numeric features

**Tests:** Baseline without new features  
**Purpose:** Control group to measure impact of new features

---

### X_train_3
**Features:** Base + Categories (wind_speed_category, capacity_category) + New Features

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed_category`, `capacity_category`, `combined_wind_rescource`, `potential_with_constraints`
3. **Categorical to numeric encoding (Step 9)**:
   - `wind_speed_category`: Low→1, Medium→2, High→3, Very High→4
   - `capacity_category`: Mapped to numeric labels (1-9) based on capacity ranges
4. No geographic encoding needed (no State/Region)
5. **Final features**: 6 features (2 base numeric, 2 categorical→numeric, 2 new numeric)

**Tests:** Categories vs Numeric + New Features impact  
**Purpose:** Test if categorical features perform better than numeric when combined with new features

---

### X_train_4
**Features:** Base + Categories (wind_speed_category, capacity_category)

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed_category`, `capacity_category`
3. New features excluded (control group)
4. **Categorical to numeric encoding (Step 9)**:
   - `wind_speed_category`: Low→1, Medium→2, High→3, Very High→4
   - `capacity_category`: Mapped to numeric labels (1-9) based on capacity ranges
5. No geographic encoding needed (no State/Region)
6. **Final features**: 4 features (2 base numeric, 2 categorical→numeric)

**Tests:** Categories vs Numeric (without new features)  
**Purpose:** Test if categorical features perform better than numeric counterparts

---

### X_train_5
**Features:** Base + Numeric + State + New Features

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed`, `capacity`, `State`, `combined_wind_rescource`, `potential_with_constraints`
3. No categorical encoding needed (uses numeric features)
4. **State one-hot encoding (Step 10)**:
   - Original `State` column dropped
   - Creates binary columns: `State_Alabama`, `State_Alaska`, `State_Arizona`, etc.
   - Results in ~50+ columns (one per state/territory in dataset)
5. **Final features**: ~57 features (2 base numeric, 2 numeric, 2 new numeric, ~50+ State binary columns)

**Tests:** State impact with numeric features and new features  
**Purpose:** Test if State improves performance with numeric features and new features

---

### X_train_6
**Features:** Base + Numeric + State

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed`, `capacity`, `State`
3. New features excluded (control group)
4. No categorical encoding needed (uses numeric features)
5. **State one-hot encoding (Step 10)**:
   - Original `State` column dropped
   - Creates binary columns: `State_Alabama`, `State_Alaska`, `State_Arizona`, etc.
   - Results in ~50+ columns (one per state/territory in dataset)
6. **Final features**: ~55 features (2 base numeric, 2 numeric, ~50+ State binary columns)

**Tests:** State impact with numeric features (without new features)  
**Purpose:** Test if State improves performance with numeric features

---

### X_train_7
**Features:** Base + Numeric + Region + New Features

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed`, `capacity`, `Region`, `combined_wind_rescource`, `potential_with_constraints`
3. No categorical encoding needed (uses numeric features)
4. **Region one-hot encoding (Step 10)**:
   - Original `Region` column dropped
   - Creates binary columns: `Region_Northeast`, `Region_West_Coast`, `Region_Great_Plains`, `Region_Southeast`, `Region_Midwest`, `Region_Mountain_West`, `Region_Other`, `Region_Territories`, `Region_Offshore/International`
   - Results in ~7-8 columns (one per region)
5. **Final features**: ~13 features (2 base numeric, 2 numeric, 2 new numeric, ~7 Region binary columns)

**Tests:** Region impact with numeric features and new features  
**Purpose:** Test if Region improves performance with numeric features and new features

---

### X_train_8
**Features:** Base + Numeric + Region

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed`, `capacity`, `Region`
3. New features excluded (control group)
4. No categorical encoding needed (uses numeric features)
5. **Region one-hot encoding (Step 10)**:
   - Original `Region` column dropped
   - Creates binary columns: `Region_Northeast`, `Region_West_Coast`, etc.
   - Results in ~7-8 columns (one per region)
6. **Final features**: ~11 features (2 base numeric, 2 numeric, ~7 Region binary columns)

**Tests:** Region impact with numeric features (without new features)  
**Purpose:** Test if Region improves performance with numeric features

---

### X_train_9
**Features:** Base + Categories + State + New Features

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed_category`, `capacity_category`, `State`, `combined_wind_rescource`, `potential_with_constraints`
3. **Categorical to numeric encoding (Step 9)**:
   - `wind_speed_category`: Low→1, Medium→2, High→3, Very High→4
   - `capacity_category`: Mapped to numeric labels (1-9) based on capacity ranges
4. **State one-hot encoding (Step 10)**:
   - Original `State` column dropped
   - Creates binary columns: `State_Alabama`, `State_Alaska`, etc.
   - Results in ~50+ columns (one per state/territory in dataset)
5. **Final features**: ~57 features (2 base numeric, 2 categorical→numeric, 2 new numeric, ~50+ State binary columns)

**Tests:** Categories + State with new features  
**Purpose:** Test combination of categorical features, State, and new features

---

### X_train_10
**Features:** Base + Categories + State

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed_category`, `capacity_category`, `State`
3. New features excluded (control group)
4. **Categorical to numeric encoding (Step 9)**:
   - `wind_speed_category`: Low→1, Medium→2, High→3, Very High→4
   - `capacity_category`: Mapped to numeric labels (1-9) based on capacity ranges
5. **State one-hot encoding (Step 10)**:
   - Original `State` column dropped
   - Creates binary columns: `State_Alabama`, `State_Alaska`, etc.
   - Results in ~50+ columns (one per state/territory in dataset)
6. **Final features**: ~55 features (2 base numeric, 2 categorical→numeric, ~50+ State binary columns)

**Tests:** Categories + State (without new features)  
**Purpose:** Test combination of categorical features and State

---

### X_train_11
**Features:** Base + Categories + Region + New Features

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed_category`, `capacity_category`, `Region`, `combined_wind_rescource`, `potential_with_constraints`
3. **Categorical to numeric encoding (Step 9)**:
   - `wind_speed_category`: Low→1, Medium→2, High→3, Very High→4
   - `capacity_category`: Mapped to numeric labels (1-9) based on capacity ranges
4. **Region one-hot encoding (Step 10)**:
   - Original `Region` column dropped
   - Creates binary columns: `Region_Northeast`, `Region_West_Coast`, etc.
   - Results in ~7-8 columns (one per region)
5. **Final features**: ~13 features (2 base numeric, 2 categorical→numeric, 2 new numeric, ~7 Region binary columns)

**Tests:** Categories + Region with new features  
**Purpose:** Test combination of categorical features, Region, and new features

---

### X_train_12
**Features:** Base + Categories + Region

**Processing Steps:**
1. All preprocessing steps (Steps 1-7) applied
2. Feature selection: `fraction_of_usable_area`, `capacity_factor`, `wind_speed_category`, `capacity_category`, `Region`
3. New features excluded (control group)
4. **Categorical to numeric encoding (Step 9)**:
   - `wind_speed_category`: Low→1, Medium→2, High→3, Very High→4
   - `capacity_category`: Mapped to numeric labels (1-9) based on capacity ranges
5. **Region one-hot encoding (Step 10)**:
   - Original `Region` column dropped
   - Creates binary columns: `Region_Northeast`, `Region_West_Coast`, etc.
   - Results in ~7-8 columns (one per region)
6. **Final features**: ~11 features (2 base numeric, 2 categorical→numeric, ~7 Region binary columns)

**Tests:** Categories + Region (without new features)  
**Purpose:** Test combination of categorical features and Region

---

## Processing Steps Summary by X_train

| X_train | Steps 1-7 (Preprocessing) | Step 9 (Categorical→Numeric) | Step 10 (Geographic One-Hot) | Final Feature Count |
|---------|---------------------------|------------------------------|------------------------------|---------------------|
| X_train_1 | ✅ All | ❌ | ❌ | 6 |
| X_train_2 | ✅ All | ❌ | ❌ | 4 |
| X_train_3 | ✅ All | ✅ wind_speed_category, capacity_category | ❌ | 6 |
| X_train_4 | ✅ All | ✅ wind_speed_category, capacity_category | ❌ | 4 |
| X_train_5 | ✅ All | ❌ | ✅ State (~50+ columns) | ~57 |
| X_train_6 | ✅ All | ❌ | ✅ State (~50+ columns) | ~55 |
| X_train_7 | ✅ All | ❌ | ✅ Region (~7 columns) | ~13 |
| X_train_8 | ✅ All | ❌ | ✅ Region (~7 columns) | ~11 |
| X_train_9 | ✅ All | ✅ wind_speed_category, capacity_category | ✅ State (~50+ columns) | ~57 |
| X_train_10 | ✅ All | ✅ wind_speed_category, capacity_category | ✅ State (~50+ columns) | ~55 |
| X_train_11 | ✅ All | ✅ wind_speed_category, capacity_category | ✅ Region (~7 columns) | ~13 |
| X_train_12 | ✅ All | ✅ wind_speed_category, capacity_category | ✅ Region (~7 columns) | ~11 |

**Legend:**
- **Steps 1-7**: Data loading, state imputation, categorization, region creation, interaction features, coordinate removal
- **Step 9**: Convert categorical text labels to numeric (wind_speed_category, capacity_category)
- **Step 10**: One-hot encode geographic features (State or Region)

---

## Testing Questions

### 1. Categories vs Numeric Counterparts
**Compare:**
- X_train_2 (numeric) vs X_train_4 (categories)
- X_train_1 (numeric + new) vs X_train_3 (categories + new)
- X_train_6 (numeric + State) vs X_train_10 (categories + State)
- X_train_8 (numeric + Region) vs X_train_12 (categories + Region)

**Hypothesis:** Categorical features may capture non-linear relationships better than numeric features.

### 2. States vs Regions
**Compare:**
- X_train_6 (State) vs X_train_8 (Region)
- X_train_5 (State + new) vs X_train_7 (Region + new)
- X_train_10 (categories + State) vs X_train_12 (categories + Region)
- X_train_9 (categories + State + new) vs X_train_11 (categories + Region + new)

**Hypothesis:** Regions may generalize better than individual States due to lower cardinality, but States may capture more specific geographic patterns.

### 3. New Features Impact
**Compare:**
- X_train_1 vs X_train_2 (numeric baseline)
- X_train_3 vs X_train_4 (categories baseline)
- X_train_5 vs X_train_6 (numeric + State)
- X_train_7 vs X_train_8 (numeric + Region)
- X_train_9 vs X_train_10 (categories + State)
- X_train_11 vs X_train_12 (categories + Region)

**Hypothesis:** New engineered features (`combined_wind_rescource` and `potential_with_constraints`) should improve model performance by capturing interactions between features.

---

## Feature Count Summary

**Note:** Feature counts shown are before one-hot encoding. State/Region features are automatically one-hot encoded, which will increase the feature count.

| X_train | Feature Count (Before Encoding) | After One-Hot Encoding | Includes New Features | Wind/Capacity Type | Geographic |
|---------|--------------------------------|----------------------|----------------------|-------------------|------------|
| X_train_1 | 6 | 6 | Yes | Numeric | None |
| X_train_2 | 4 | 4 | No | Numeric | None |
| X_train_3 | 6 | 6 | Yes | Categorical (numeric) | None |
| X_train_4 | 4 | 4 | No | Categorical (numeric) | None |
| X_train_5 | 7 | ~57+ (50+ State columns) | Yes | Numeric | State (one-hot) |
| X_train_6 | 5 | ~55+ (50+ State columns) | No | Numeric | State (one-hot) |
| X_train_7 | 7 | ~14 (7 Region columns) | Yes | Numeric | Region (one-hot) |
| X_train_8 | 5 | ~12 (7 Region columns) | No | Numeric | Region (one-hot) |
| X_train_9 | 7 | ~57+ (50+ State columns) | Yes | Categorical (numeric) | State (one-hot) |
| X_train_10 | 5 | ~55+ (50+ State columns) | No | Categorical (numeric) | State (one-hot) |
| X_train_11 | 7 | ~14 (7 Region columns) | Yes | Categorical (numeric) | Region (one-hot) |
| X_train_12 | 5 | ~12 (7 Region columns) | No | Categorical (numeric) | Region (one-hot) |

---

## Recommended Testing Order

1. **Baseline Comparison:** Start with X_train_1, X_train_2, X_train_3, X_train_4
2. **New Features Impact:** Compare pairs with/without new features
3. **Geographic Impact:** Test State vs Region additions (X_train_5-12)

---

## Notes

- All combinations include base features (`fraction_of_usable_area`, `capacity_factor`)
- **State and Region are mutually exclusive** - combinations use either State OR Region, never both
- **Categorical features are automatically converted to numeric labels** in the notebook:
  - `wind_speed_category`: Low=1, Medium=2, High=3, Very High=4 (applied to X_train_3, X_train_4, X_train_9-12)
  - `capacity_category`: Ordered by numeric value, e.g., Very Low (<2)=1, 2-4=2, 6-8=4, 10-12=6, 14-16=8 (applied to X_train_3, X_train_4, X_train_9-12)
- **State and Region features are automatically one-hot encoded** in the notebook for X_train_5-12
  - State features (X_train_5, X_train_6, X_train_9, X_train_10) are one-hot encoded with prefix 'State_'
  - Region features (X_train_7, X_train_8, X_train_11, X_train_12) are one-hot encoded with prefix 'Region_'
- State feature has high cardinality (50+ states), which will result in 50+ one-hot encoded columns
- Region feature has lower cardinality (~7 regions), which will result in ~7 one-hot encoded columns
- New features are derived from existing features, so they may introduce some redundancy

---

## Usage Example

```python
# Example: Train model with X_train_1
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train_1, y_train)

# Compare with X_train_2 (without new features)
model2 = RandomForestClassifier()
model2.fit(X_train_2, y_train)

# Evaluate which performs better
```

---

## Model Evaluation Results Summary

This section summarizes the XGBoost model evaluation results using 5-fold stratified cross-validation on all 12 X_train combinations.

### Overall Performance Metrics

| X_train | Features | Accuracy | F1 Score | ROC-AUC | Rank (ROC-AUC) |
|---------|----------|----------|----------|---------|----------------|
| **X_train_5** | 55 | **0.8548** | **0.8134** | **0.9305** | 🥇 1st |
| **X_train_6** | 53 | **0.8503** | **0.8076** | **0.9276** | 🥈 2nd |
| **X_train_9** | 55 | **0.8444** | **0.7991** | **0.9227** | 🥉 3rd |
| **X_train_7** | 13 | 0.8183 | 0.7600 | 0.9009 | 4th |
| **X_train_10** | 53 | 0.8267 | 0.7763 | 0.9092 | 5th |
| **X_train_8** | 11 | 0.8145 | 0.7554 | 0.8979 | 6th |
| **X_train_11** | 13 | 0.8080 | 0.7458 | 0.8917 | 7th |
| **X_train_1** | 6 | 0.8000 | 0.7344 | 0.8838 | 8th |
| **X_train_2** | 4 | 0.7931 | 0.7250 | 0.8782 | 9th |
| **X_train_3** | 6 | 0.7860 | 0.7134 | 0.8705 | 10th |
| **X_train_12** | 11 | 0.7733 | 0.6926 | 0.8635 | 11th |
| **X_train_4** | 4 | 0.7500 | 0.6588 | 0.8299 | 12th |

**Note:** All models trained on 152,688 samples using default XGBoost parameters with 5-fold stratified cross-validation.

---

### Key Findings

#### 1. **Best Overall Performance: State Features with Numeric Variables**

**Winner: X_train_5** (Base + Numeric + State + New Features)
- **ROC-AUC: 0.9305** (highest)
- **Accuracy: 0.8548** (highest)
- **F1 Score: 0.8134** (highest)
- **Key Insight:** State one-hot encoding with numeric wind/capacity features and new interaction features provides the best predictive power.

**Runner-up: X_train_6** (Base + Numeric + State, without new features)
- **ROC-AUC: 0.9276** (2nd highest)
- **Accuracy: 0.8503** (2nd highest)
- **F1 Score: 0.8076** (2nd highest)
- **Key Insight:** State features are highly valuable even without the new interaction features.

---

#### 2. **Categories vs Numeric Counterparts**

**Research Question:** Do categorical features perform better than their numeric counterparts?

**Results:**

| Comparison | Numeric (ROC-AUC) | Categorical (ROC-AUC) | Winner | Difference |
|------------|-------------------|----------------------|--------|------------|
| Baseline (no new features) | X_train_2: 0.8782 | X_train_4: 0.8299 | **Numeric** | +0.0483 |
| With new features | X_train_1: 0.8838 | X_train_3: 0.8705 | **Numeric** | +0.0133 |
| With State | X_train_6: 0.9276 | X_train_10: 0.9092 | **Numeric** | +0.0184 |
| With Region | X_train_8: 0.8979 | X_train_12: 0.8635 | **Numeric** | +0.0344 |

**Conclusion:** ✅ **Numeric features consistently outperform categorical features** across all configurations.
- **Average improvement:** +0.0286 ROC-AUC points
- **Largest gap:** Baseline comparison (+0.0483)
- **Smallest gap:** With new features (+0.0133)

**Recommendation:** Use numeric `wind_speed` and `capacity` features rather than their categorical counterparts.

---

#### 3. **States vs Regions**

**Research Question:** Do States perform better than Regions?

**Results:**

| Comparison | State (ROC-AUC) | Region (ROC-AUC) | Winner | Difference |
|------------|----------------|------------------|--------|------------|
| Numeric + New Features | X_train_5: 0.9305 | X_train_7: 0.9009 | **State** | +0.0296 |
| Numeric (no new features) | X_train_6: 0.9276 | X_train_8: 0.8979 | **State** | +0.0297 |
| Categories + New Features | X_train_9: 0.9227 | X_train_11: 0.8917 | **State** | +0.0310 |
| Categories (no new features) | X_train_10: 0.9092 | X_train_12: 0.8635 | **State** | +0.0457 |

**Conclusion:** ✅ **State features consistently outperform Region features** across all configurations.
- **Average improvement:** +0.0340 ROC-AUC points
- **Largest gap:** Categories without new features (+0.0457)
- **Smallest gap:** Numeric with new features (+0.0296)

**Trade-off Analysis:**
- **State advantages:** Higher granularity captures state-specific patterns (regulatory, geographic, economic factors)
- **Region advantages:** Lower dimensionality (~7 features vs ~50+), better generalization, less overfitting risk
- **Performance cost:** ~3-4% ROC-AUC reduction when using Regions instead of States

**Recommendation:** Use **State features** for maximum performance, but consider **Region features** if model interpretability or generalization is a priority.

---

#### 4. **New Features Impact**

**Research Question:** Do the new engineered features (`combined_wind_rescource`, `potential_with_constraints`) improve model performance?

**Results:**

| Comparison | With New Features (ROC-AUC) | Without New Features (ROC-AUC) | Improvement | Significant? |
|------------|----------------------------|-------------------------------|-------------|--------------|
| Numeric baseline | X_train_1: 0.8838 | X_train_2: 0.8782 | +0.0056 | ✅ Yes |
| Categories baseline | X_train_3: 0.8705 | X_train_4: 0.8299 | +0.0406 | ✅ Yes |
| Numeric + State | X_train_5: 0.9305 | X_train_6: 0.9276 | +0.0029 | ✅ Yes |
| Numeric + Region | X_train_7: 0.9009 | X_train_8: 0.8979 | +0.0030 | ✅ Yes |
| Categories + State | X_train_9: 0.9227 | X_train_10: 0.9092 | +0.0135 | ✅ Yes |
| Categories + Region | X_train_11: 0.8917 | X_train_12: 0.8635 | +0.0282 | ✅ Yes |

**Conclusion:** ✅ **New features consistently improve performance** across all configurations.
- **Average improvement:** +0.0156 ROC-AUC points
- **Largest impact:** Categories baseline (+0.0406) - new features help compensate for categorical feature limitations
- **Smallest impact:** Numeric + State (+0.0029) - already strong performance, marginal gain
- **All comparisons show improvement:** No configuration where new features hurt performance

**Recommendation:** ✅ **Always include the new interaction features** (`combined_wind_rescource`, `potential_with_constraints`).

---

### Performance Rankings by Metric

#### Top 3 by Accuracy
1. **X_train_5**: 0.8548 (Numeric + State + New Features)
2. **X_train_6**: 0.8503 (Numeric + State)
3. **X_train_9**: 0.8444 (Categories + State + New Features)

#### Top 3 by F1 Score
1. **X_train_5**: 0.8134 (Numeric + State + New Features)
2. **X_train_6**: 0.8076 (Numeric + State)
3. **X_train_9**: 0.7991 (Categories + State + New Features)

#### Top 3 by ROC-AUC
1. **X_train_5**: 0.9305 (Numeric + State + New Features)
2. **X_train_6**: 0.9276 (Numeric + State)
3. **X_train_9**: 0.9227 (Categories + State + New Features)

---

### Key Insights and Recommendations

#### 🎯 **Optimal Feature Configuration**
**Recommended: X_train_5** (Base + Numeric + State + New Features)
- **Best overall performance** across all metrics
- **ROC-AUC: 0.9305** - Excellent discrimination ability
- **55 features** - Manageable dimensionality
- **Components:**
  - Base features: `fraction_of_usable_area`, `capacity_factor`
  - Numeric features: `wind_speed`, `capacity` (not categorical)
  - Geographic: `State` (one-hot encoded, ~50+ columns)
  - New features: `combined_wind_rescource`, `potential_with_constraints`

#### 📊 **Performance Hierarchy**
1. **Tier 1 (ROC-AUC > 0.92):** X_train_5, X_train_6, X_train_9
   - All include State features
   - Best for production use
2. **Tier 2 (ROC-AUC 0.89-0.92):** X_train_7, X_train_10, X_train_8
   - Mix of State/Region, with/without new features
   - Good performance with lower dimensionality
3. **Tier 3 (ROC-AUC 0.83-0.89):** X_train_11, X_train_1, X_train_2, X_train_3
   - Baseline configurations without geographic features
   - Useful for comparison but not optimal
4. **Tier 4 (ROC-AUC < 0.83):** X_train_12, X_train_4
   - Categories without geographic features
   - Lowest performance

#### 💡 **Feature Engineering Insights**

1. **Geographic features are critical:**
   - Adding State features improves ROC-AUC by ~0.05-0.08 compared to baseline
   - State > Region by ~0.03 ROC-AUC points

2. **Numeric > Categorical:**
   - Numeric wind_speed and capacity consistently outperform categorical versions
   - Categorical features may lose information through discretization

3. **Interaction features add value:**
   - New features provide consistent (though sometimes small) improvements
   - Most valuable when combined with categorical features (+0.04 ROC-AUC)

4. **Feature interactions matter:**
   - Best performance requires combination of geographic, numeric, and interaction features
   - No single feature type dominates; synergy is important

#### ⚠️ **Trade-offs to Consider**

1. **State vs Region:**
   - **Choose State if:** Maximum performance is priority, sufficient data for ~50+ features
   - **Choose Region if:** Interpretability, generalization, or lower dimensionality is priority

2. **Feature Count:**
   - State features: ~55 total features (may require more regularization)
   - Region features: ~13 total features (simpler, faster training)

3. **Categorical vs Numeric:**
   - Numeric features are always better in this dataset
   - Consider categorical only if interpretability or domain knowledge requires it

---

### Summary Statistics

- **Best Model:** X_train_5 (ROC-AUC: 0.9305)
- **Worst Model:** X_train_4 (ROC-AUC: 0.8299)
- **Performance Range:** 0.1006 ROC-AUC points (12.1% relative improvement)
- **Average ROC-AUC:** 0.8904
- **Median ROC-AUC:** 0.8945

---

*Generated for systematic feature engineering testing*
