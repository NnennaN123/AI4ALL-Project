"""
Wind Turbine Location Prediction - Interactive Demo
CRANBerry Team - AI4ALL Project

This Streamlit app allows users to interactively test our wind turbine prediction models.
Users can input site characteristics or select from real locations to see predictions.

UPDATED: Now includes XGBoost model with feature engineering for best-in-class performance!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Wind Turbine Site Predictor | CRANBerry Team",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .turbine-yes {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .turbine-no {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .confidence-high {
        color: #10B981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .confidence-low {
        color: #EF4444;
        font-weight: bold;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .badge-best {
        background-color: #10B981;
        color: white;
    }
    .badge-good {
        background-color: #F59E0B;
        color: white;
    }
    .badge-baseline {
        background-color: #6B7280;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load models and metrics
@st.cache_resource
def load_models_and_metrics():
    """Load trained models, scaler, and metrics from files."""
    try:
        # Get the directory where this script is located
        BASE_DIR = Path(__file__).parent
        
        # Load models with absolute paths
        log_reg = joblib.load(BASE_DIR / 'logistic_regression_wind_model.pkl')
        rf_model = joblib.load(BASE_DIR / 'random_forest_wind_model.pkl')
        scaler = joblib.load(BASE_DIR / 'scaler.pkl')
        
        # Try to load XGBoost model (may not exist in older versions)
        try:
            xgb_model = joblib.load(BASE_DIR / 'xgboost_tuned_feat_eng_wind_model.pkl')
            xgb_available = True
        except FileNotFoundError:
            st.warning("⚠️ XGBoost model not found. Using Logistic Regression and Random Forest only.")
            xgb_model = None
            xgb_available = False
        
        # Load metrics
        with open(BASE_DIR / 'model_metrics.json', 'r') as f:
            log_reg_metrics = json.load(f)
        
        with open(BASE_DIR / 'random_forest_model_metrics.json', 'r') as f:
            rf_metrics = json.load(f)
        
        # Try to load XGBoost metrics
        if xgb_available:
            try:
                with open(BASE_DIR / 'xgboost_tuned_feat_eng_model_metrics.json', 'r') as f:
                    xgb_metrics = json.load(f)
            except FileNotFoundError:
                xgb_metrics = None
                st.warning("⚠️ XGBoost metrics not found.")
        else:
            xgb_metrics = None
        
        return log_reg, rf_model, xgb_model, scaler, log_reg_metrics, rf_metrics, xgb_metrics, xgb_available
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.error(f"Looking in directory: {Path(__file__).parent}")
        st.stop()

# Load models
log_reg_model, rf_model, xgb_model, scaler, log_metrics, rf_metrics, xgb_metrics, xgb_available = load_models_and_metrics()

# US States for XGBoost model
US_STATES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
    'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 
    'Wisconsin', 'Wyoming'
]

# Sample locations with real characteristics (updated with state info)
SAMPLE_LOCATIONS = {
    "🌟 West Texas (High Potential)": {
        "latitude": 31.5,
        "longitude": -102.0,
        "fraction_of_usable_area": 0.85,
        "capacity": 2500,
        "wind_speed": 8.2,
        "capacity_factor": 0.42,
        "state": "Texas",
        "description": "Sweetwater Wind Farm region - known for strong, consistent winds",
        "actual": "Has turbines"
    },
    "🌾 Iowa Plains (Good)": {
        "latitude": 42.5,
        "longitude": -93.5,
        "fraction_of_usable_area": 0.90,
        "capacity": 2000,
        "wind_speed": 7.5,
        "capacity_factor": 0.38,
        "state": "Iowa",
        "description": "Iowa wind corridor - agricultural area with good wind resources",
        "actual": "Has turbines"
    },
    "🌊 California Coast (Moderate)": {
        "latitude": 35.0,
        "longitude": -120.5,
        "fraction_of_usable_area": 0.60,
        "capacity": 1800,
        "wind_speed": 7.0,
        "capacity_factor": 0.32,
        "state": "California",
        "description": "Coastal region - seasonal winds with terrain challenges",
        "actual": "Some turbines"
    },
    "⚡ Oklahoma Panhandle (Good)": {
        "latitude": 36.5,
        "longitude": -100.5,
        "fraction_of_usable_area": 0.80,
        "capacity": 2200,
        "wind_speed": 8.0,
        "capacity_factor": 0.40,
        "state": "Oklahoma",
        "description": "Great Plains - flat terrain with strong winds",
        "actual": "Has turbines"
    },
    "🏖️ Florida (Poor)": {
        "latitude": 28.5,
        "longitude": -81.5,
        "fraction_of_usable_area": 0.40,
        "capacity": 1200,
        "wind_speed": 5.5,
        "capacity_factor": 0.20,
        "state": "Florida",
        "description": "Low wind resource - not suitable for utility-scale wind",
        "actual": "No turbines"
    },
    "🏔️ Montana Plains (Excellent)": {
        "latitude": 47.5,
        "longitude": -109.5,
        "fraction_of_usable_area": 0.95,
        "capacity": 3000,
        "wind_speed": 8.5,
        "capacity_factor": 0.45,
        "state": "Montana",
        "description": "High plains with excellent wind resources",
        "actual": "Has turbines"
    }
}

def get_confidence_level(probability):
    """Determine confidence level and styling based on probability."""
    if probability > 0.7 or probability < 0.3:
        return "High", "confidence-high"
    elif probability > 0.55 or probability < 0.45:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def prepare_xgb_features(fraction_usable, capacity, wind_speed, capacity_factor, state):
    """Prepare features for XGBoost model with feature engineering and state encoding."""
    # Base features
    features_dict = {
        'fraction_of_usable_area': fraction_usable,
        'capacity_factor': capacity_factor,
        'wind_speed': wind_speed,
        'capacity': capacity,
    }
    
    # Engineered features
    features_dict['combined_wind_rescource'] = wind_speed * capacity_factor
    features_dict['potential_with_constraints'] = capacity * fraction_usable
    
    # One-hot encode state
    for us_state in US_STATES:
        features_dict[f'State_{us_state}'] = 1 if state == us_state else 0
    
    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([features_dict])
    
    # Ensure columns are in the correct order expected by the model
    expected_cols = ['fraction_of_usable_area', 'capacity_factor', 'wind_speed', 'capacity',
                     'combined_wind_rescource', 'potential_with_constraints'] + [f'State_{s}' for s in US_STATES]
    
    # Reorder columns to match training order
    feature_df = feature_df[expected_cols]
    
    return feature_df

def make_prediction(features_array, model, use_scaler=False):
    """Make prediction with given model."""
    if use_scaler:
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
    else:
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
    
    return prediction, probability

def create_probability_gauge(probability, title):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 50, 'increasing': {'color': "#667eea"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#f5576c'},
                {'range': [30, 70], 'color': '#ffd93d'},
                {'range': [70, 100], 'color': '#10B981'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'size': 14}
    )
    
    return fig

def create_feature_importance_chart(model_type='rf'):
    """Create feature importance visualization."""
    if model_type == 'rf':
        feature_names = ['Fraction Usable Area', 'Capacity', 'Wind Speed', 'Capacity Factor']
        importances = rf_model.feature_importances_
    else:  # XGBoost
        if xgb_model is None:
            return None
        # For XGBoost, get feature importance and show top features
        importances = xgb_model.feature_importances_
        feature_names = ['Fraction Usable Area', 'Capacity Factor', 'Wind Speed', 'Capacity',
                        'Combined Wind Resource', 'Potential with Constraints'] + [f'State_{s}' for s in US_STATES]
        
        # Get indices of top 10 features
        indices = np.argsort(importances)[::-1][:10]
        importances = importances[indices]
        feature_names = [feature_names[i] for i in indices]
    
    # Sort by importance
    if model_type == 'rf':
        indices = np.argsort(importances)[::-1]
        importances = importances[indices]
        feature_names = [feature_names[i] for i in indices]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=feature_names,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{imp:.1%}' for imp in importances],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Feature Importance - {"Random Forest" if model_type == "rf" else "XGBoost (Top 10)"}',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=400 if model_type == 'rf' else 500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# ============================================================================
# MAIN APP LAYOUT
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🌬️ Wind Turbine Site Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Location Prediction by Team CRANBerry</p>', unsafe_allow_html=True)

# Model performance banner
if xgb_available and xgb_metrics:
    st.success(f"""
    ⭐ **Latest Update:** Now featuring our best-performing XGBoost model!
    - **ROC-AUC: {xgb_metrics['roc_auc']:.4f}** (95.5% accuracy in ranking)
    - **Accuracy: {xgb_metrics['classification_report']['accuracy']:.4f}** (87.8% correct predictions)
    - **Turbine Detection: {xgb_metrics['classification_report']['True']['recall']:.1%}** (94.3% of turbine sites found)
    """)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Make Predictions", 
    "📊 Model Performance", 
    "🔍 Model Comparison",
    "📖 About Project"
])

# TAB 1: Make Predictions
with tab1:
    st.header("Predict Wind Turbine Suitability")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Use Sample Location", "Enter Custom Values"],
        horizontal=True
    )
    
    if input_method == "Use Sample Location":
        selected_location = st.selectbox(
            "Select a sample location:",
            list(SAMPLE_LOCATIONS.keys())
        )
        
        loc_data = SAMPLE_LOCATIONS[selected_location]
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>{selected_location}</h4>
            <p><strong>Description:</strong> {loc_data['description']}</p>
            <p><strong>Actual Status:</strong> {loc_data['actual']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display location characteristics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", f"{loc_data['latitude']:.2f}°")
            st.metric("Fraction Usable Area", f"{loc_data['fraction_of_usable_area']:.2%}")
            st.metric("Wind Speed", f"{loc_data['wind_speed']} m/s")
        with col2:
            st.metric("Longitude", f"{loc_data['longitude']:.2f}°")
            st.metric("Capacity", f"{loc_data['capacity']:,} kW")
            st.metric("Capacity Factor", f"{loc_data['capacity_factor']:.2%}")
        if xgb_available:
            st.metric("State", loc_data['state'])
        
        # Use location data for prediction
        fraction_usable = loc_data['fraction_of_usable_area']
        capacity = loc_data['capacity']
        wind_speed = loc_data['wind_speed']
        capacity_factor = loc_data['capacity_factor']
        state = loc_data.get('state', 'Texas')  # Default to Texas if not specified
        
    else:  # Custom input
        st.markdown("### Enter Site Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fraction_usable = st.slider(
                "Fraction of Usable Area",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="Proportion of grid cell usable for wind development (0-1)"
            )
            
            capacity = st.number_input(
                "Potential Capacity (kW)",
                min_value=0,
                max_value=5000,
                value=2000,
                step=100,
                help="Maximum power generation capacity"
            )
            
            wind_speed = st.slider(
                "Average Wind Speed (m/s)",
                min_value=0.0,
                max_value=15.0,
                value=7.5,
                step=0.5,
                help="Mean wind speed at hub height"
            )
        
        with col2:
            capacity_factor = st.slider(
                "Capacity Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.35,
                step=0.05,
                help="Expected efficiency (actual output / maximum output)"
            )
            
            if xgb_available:
                state = st.selectbox(
                    "State",
                    US_STATES,
                    index=US_STATES.index('Texas'),
                    help="US State for geographic context (XGBoost only)"
                )
            else:
                state = "Texas"  # Default
    
    st.markdown("---")
    
    # Model selection
    if xgb_available:
        model_choice = st.radio(
            "Select prediction model:",
            ["XGBoost (Best) ⭐", "Logistic Regression (Fast)", "Random Forest (Balanced)"],
            horizontal=True
        )
    else:
        model_choice = st.radio(
            "Select prediction model:",
            ["Logistic Regression (Fast)", "Random Forest (Balanced)"],
            horizontal=True
        )
    
    # Prepare features
    features_basic = np.array([[fraction_usable, capacity, wind_speed, capacity_factor]])
    
    # Make predictions based on selected model
    if "XGBoost" in model_choice and xgb_available:
        features_xgb = prepare_xgb_features(fraction_usable, capacity, wind_speed, capacity_factor, state)
        prediction, probability = make_prediction(features_xgb, xgb_model, use_scaler=False)
        model_name = "XGBoost"
        model_info = "🌟 Best overall performance with feature engineering"
    elif "Logistic" in model_choice:
        prediction, probability = make_prediction(features_basic, log_reg_model, use_scaler=True)
        model_name = "Logistic Regression"
        model_info = "⚡ Fast, interpretable baseline model"
    else:  # Random Forest
        prediction, probability = make_prediction(features_basic, rf_model, use_scaler=False)
        model_name = "Random Forest"
        model_info = "🎯 Balanced precision and recall"
    
    # Display prediction
    st.markdown("### Prediction Results")
    st.info(f"**Model Used:** {model_name} - {model_info}")
    
    turbine_prob = probability[1]
    no_turbine_prob = probability[0]
    
    # Main prediction display
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-box turbine-yes">
            ✅ TURBINE LIKELY
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box turbine-no">
            ❌ TURBINE UNLIKELY
        </div>
        """, unsafe_allow_html=True)
    
    # Probability gauges
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_probability_gauge(turbine_prob, "Turbine Probability"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_probability_gauge(no_turbine_prob, "No Turbine Probability"),
            use_container_width=True
        )
    
    # Confidence level
    confidence, conf_class = get_confidence_level(turbine_prob)
    st.markdown(f"**Model Confidence:** <span class='{conf_class}'>{confidence}</span> ({turbine_prob:.1%})", 
                unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("### 💡 Interpretation")
    if turbine_prob > 0.7:
        st.success(f"""
        **High Confidence Positive:** This site shows strong indicators for wind turbine development.
        The model is {turbine_prob:.1%} confident that this location is suitable for turbines based on:
        - Wind resource characteristics
        - Site capacity and usable area
        {"- Geographic factors (state)" if "XGBoost" in model_choice else ""}
        """)
    elif turbine_prob > 0.5:
        st.warning(f"""
        **Moderate Confidence Positive:** This site shows some potential for wind development.
        The model gives {turbine_prob:.1%} confidence, suggesting further investigation may be warranted.
        Consider additional factors like transmission access and environmental constraints.
        """)
    elif turbine_prob > 0.3:
        st.info(f"""
        **Uncertain:** The model shows mixed signals for this location ({turbine_prob:.1%} probability).
        This borderline case would benefit from:
        - Detailed on-site wind measurements
        - Economic feasibility analysis
        - Expert review of local conditions
        """)
    else:
        st.error(f"""
        **High Confidence Negative:** This site is unlikely suitable for wind turbine development.
        The model is {no_turbine_prob:.1%} confident this location lacks the necessary characteristics,
        likely due to insufficient wind resources or site constraints.
        """)

# TAB 2: Model Performance
with tab2:
    st.header("Model Performance Metrics")
    
    # Model selector
    if xgb_available:
        perf_model = st.selectbox(
            "Select model to view:",
            ["XGBoost (Best) ⭐", "Random Forest", "Logistic Regression"]
        )
    else:
        perf_model = st.selectbox(
            "Select model to view:",
            ["Random Forest", "Logistic Regression"]
        )
    
    if "XGBoost" in perf_model and xgb_available and xgb_metrics:
        st.markdown("### 🌟 XGBoost - Production Model")
        st.success("⭐ This is our best-performing model with feature engineering and hyperparameter tuning!")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{xgb_metrics['roc_auc']:.4f}", "+18.2% vs RF")
        col2.metric("Accuracy", f"{xgb_metrics['classification_report']['accuracy']:.4f}", "+17.5% vs RF")
        col3.metric("Precision (Turbine)", f"{xgb_metrics['classification_report']['True']['precision']:.3f}", "+11.7% vs RF")
        col4.metric("Recall (Turbine)", f"{xgb_metrics['classification_report']['True']['recall']:.3f}", "+49.5% vs RF")
        
        # Classification report
        st.markdown("#### Classification Report")
        xgb_class_df = pd.DataFrame({
            'Class': ['No Turbine', 'Turbine'],
            'Precision': [
                f"{xgb_metrics['classification_report']['False']['precision']:.1%}",
                f"{xgb_metrics['classification_report']['True']['precision']:.1%}"
            ],
            'Recall': [
                f"{xgb_metrics['classification_report']['False']['recall']:.1%}",
                f"{xgb_metrics['classification_report']['True']['recall']:.1%}"
            ],
            'F1-Score': [
                f"{xgb_metrics['classification_report']['False']['f1-score']:.3f}",
                f"{xgb_metrics['classification_report']['True']['f1-score']:.3f}"
            ]
        })
        st.dataframe(xgb_class_df, hide_index=True, use_container_width=True)
        
        st.markdown("""
        **💡 Key Strengths:**
        - **Exceptional recall (94.3%)**: Finds nearly all turbine locations
        - **High precision (78.0%)**: Minimizes false positives
        - **Best ROC-AUC (0.9545)**: Superior ranking ability
        - **Feature engineering**: Uses 56 features including state encoding and interaction terms
        """)
        
        # Feature importance
        st.markdown("---")
        st.markdown("### 🎯 Top Feature Importance (Top 10)")
        xgb_fi_chart = create_feature_importance_chart('xgb')
        if xgb_fi_chart:
            st.plotly_chart(xgb_fi_chart, use_container_width=True)
            
            st.markdown("""
            **Insights:**
            - **State encoding** provides valuable geographic context
            - **Engineered features** capture complex relationships
            - **Core features** (wind speed, capacity factor) remain highly important
            """)
        
    elif "Random Forest" in perf_model:
        st.markdown("### 🎯 Random Forest Model")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{rf_metrics['roc_auc']:.3f}")
        col2.metric("Accuracy", f"{rf_metrics['classification_report']['accuracy']:.3f}")
        col3.metric("Precision (Turbine)", f"{rf_metrics['classification_report']['True']['precision']:.3f}")
        col4.metric("Recall (Turbine)", f"{rf_metrics['classification_report']['True']['recall']:.3f}")
        
        # Classification report
        st.markdown("#### Classification Report")
        rf_class_df = pd.DataFrame({
            'Class': ['No Turbine', 'Turbine'],
            'Precision': [
                f"{rf_metrics['classification_report']['False']['precision']:.1%}",
                f"{rf_metrics['classification_report']['True']['precision']:.1%}"
            ],
            'Recall': [
                f"{rf_metrics['classification_report']['False']['recall']:.1%}",
                f"{rf_metrics['classification_report']['True']['recall']:.1%}"
            ],
            'F1-Score': [
                f"{rf_metrics['classification_report']['False']['f1-score']:.3f}",
                f"{rf_metrics['classification_report']['True']['f1-score']:.3f}"
            ]
        })
        st.dataframe(rf_class_df, hide_index=True, use_container_width=True)
        
        st.markdown("**💡 Key Strength:** Higher precision (66.3%) - better for final site validation to minimize false positives")
        
        # Feature importance
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance")
        st.plotly_chart(create_feature_importance_chart('rf'), use_container_width=True)
    
    else:  # Logistic Regression
        st.markdown("### ⚡ Logistic Regression Model")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{log_metrics['roc_auc']:.3f}")
        col2.metric("Accuracy", f"{log_metrics['classification_report']['accuracy']:.3f}")
        col3.metric("Precision (Turbine)", f"{log_metrics['classification_report']['True']['precision']:.3f}")
        col4.metric("Recall (Turbine)", f"{log_metrics['classification_report']['True']['recall']:.3f}")
        
        # Classification report
        st.markdown("#### Classification Report")
        log_class_df = pd.DataFrame({
            'Class': ['No Turbine', 'Turbine'],
            'Precision': [
                f"{log_metrics['classification_report']['False']['precision']:.1%}",
                f"{log_metrics['classification_report']['True']['precision']:.1%}"
            ],
            'Recall': [
                f"{log_metrics['classification_report']['False']['recall']:.1%}",
                f"{log_metrics['classification_report']['True']['recall']:.1%}"
            ],
            'F1-Score': [
                f"{log_metrics['classification_report']['False']['f1-score']:.3f}",
                f"{log_metrics['classification_report']['True']['f1-score']:.3f}"
            ]
        })
        st.dataframe(log_class_df, hide_index=True, use_container_width=True)
        
        st.markdown("**💡 Key Strength:** Fast inference and interpretable coefficients - good for initial screening")

# TAB 3: Model Comparison
with tab3:
    st.header("Model Comparison & Selection Guide")
    
    # Performance comparison table
    st.markdown("### 📊 Performance Metrics Comparison")
    
    if xgb_available and xgb_metrics:
        comparison_data = {
            'Metric': ['ROC-AUC', 'Accuracy', 'Precision (Turbine)', 'Recall (Turbine)', 'F1-Score (Turbine)'],
            'Logistic Regression': [
                f"{log_metrics['roc_auc']:.4f}",
                f"{log_metrics['classification_report']['accuracy']:.4f}",
                f"{log_metrics['classification_report']['True']['precision']:.3f}",
                f"{log_metrics['classification_report']['True']['recall']:.3f}",
                f"{log_metrics['classification_report']['True']['f1-score']:.3f}"
            ],
            'Random Forest': [
                f"{rf_metrics['roc_auc']:.4f}",
                f"{rf_metrics['classification_report']['accuracy']:.4f}",
                f"{rf_metrics['classification_report']['True']['precision']:.3f}",
                f"{rf_metrics['classification_report']['True']['recall']:.3f}",
                f"{rf_metrics['classification_report']['True']['f1-score']:.3f}"
            ],
            'XGBoost ⭐': [
                f"{xgb_metrics['roc_auc']:.4f}",
                f"{xgb_metrics['classification_report']['accuracy']:.4f}",
                f"{xgb_metrics['classification_report']['True']['precision']:.3f}",
                f"{xgb_metrics['classification_report']['True']['recall']:.3f}",
                f"{xgb_metrics['classification_report']['True']['f1-score']:.3f}"
            ]
        }
        
        # Visual comparison
        fig_comparison = go.Figure()
        
        models = ['Logistic Regression', 'Random Forest', 'XGBoost']
        colors = ['#667eea', '#10B981', '#F59E0B']
        
        for i, model in enumerate(models):
            values = [
                float(comparison_data[model][0]),  # ROC-AUC
                float(comparison_data[model][1]),  # Accuracy
                float(comparison_data[model][2]),  # Precision
                float(comparison_data[model][3]),  # Recall
                float(comparison_data[model][4])   # F1-Score
            ]
            
            fig_comparison.add_trace(go.Bar(
                name=model,
                x=comparison_data['Metric'],
                y=values,
                marker_color=colors[i],
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison',
            yaxis_title='Score',
            barmode='group',
            height=500,
            showlegend=True,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
    else:
        comparison_data = {
            'Metric': ['ROC-AUC', 'Accuracy', 'Precision (Turbine)', 'Recall (Turbine)', 'F1-Score (Turbine)'],
            'Logistic Regression': [
                f"{log_metrics['roc_auc']:.4f}",
                f"{log_metrics['classification_report']['accuracy']:.4f}",
                f"{log_metrics['classification_report']['True']['precision']:.3f}",
                f"{log_metrics['classification_report']['True']['recall']:.3f}",
                f"{log_metrics['classification_report']['True']['f1-score']:.3f}"
            ],
            'Random Forest': [
                f"{rf_metrics['roc_auc']:.4f}",
                f"{rf_metrics['classification_report']['accuracy']:.4f}",
                f"{rf_metrics['classification_report']['True']['precision']:.3f}",
                f"{rf_metrics['classification_report']['True']['recall']:.3f}",
                f"{rf_metrics['classification_report']['True']['f1-score']:.3f}"
            ]
        }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    # Performance evolution
    if xgb_available and xgb_metrics:
        st.markdown("---")
        st.markdown("### 📈 Model Evolution Timeline")
        
        st.markdown("""
        #### Phase 1: Baseline Models
        1. **Logistic Regression** (ROC-AUC: 0.732)
           - Linear baseline with balanced class weights
           - High recall (72.8%) for finding potential sites
        
        2. **Random Forest** (ROC-AUC: 0.770)
           - +5.2% improvement in ROC-AUC
           - Better precision (66.3%) for validation
        
        3. **Initial XGBoost** (ROC-AUC: 0.847)
           - +10.0% improvement over Random Forest
           - Best baseline performance
        
        #### Phase 2: Feature Engineering & Optimization
        4. **Feature Engineering Exploration**
           - Tested 12 feature combinations
           - Added state encoding, interaction features
           - Best combination: +8.3% ROC-AUC improvement
        
        5. **Final XGBoost with Tuning ⭐** (ROC-AUC: 0.9545)
           - GridSearchCV hyperparameter optimization
           - 56 features (base + geographic + engineered)
           - **+12.7% improvement** over initial XGBoost
           - **+30.4% improvement** over baseline
        
        **Total Journey:** 0.732 → 0.9545 ROC-AUC (+30.4% improvement)
        """)
    
    # Model recommendations
    st.markdown("---")
    st.markdown("### 🎯 Model Selection Guide")
    
    if xgb_available:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### Logistic Regression
            <span class="model-badge badge-baseline">BASELINE</span>
            
            **Use When:**
            - 🔍 Initial site screening
            - ⚡ Need fast predictions
            - 📊 Want interpretable results
            - 🎯 High recall acceptable
            
            **Best For:**
            - Broad area surveys
            - Quick feasibility checks
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            #### Random Forest
            <span class="model-badge badge-good">GOOD</span>
            
            **Use When:**
            - 🎯 Balanced performance needed
            - 📈 Non-linear relationships
            - 🔬 Feature analysis needed
            - No state data available
            
            **Best For:**
            - Mid-stage evaluation
            - Portfolio optimization
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            #### XGBoost ⭐
            <span class="model-badge badge-best">BEST</span>
            
            **Use When:**
            - ✅ Maximum accuracy needed
            - 🎯 Final site selection
            - 📍 State data available
            - 💰 High-stakes decisions
            
            **Best For:**
            - Production deployment
            - Investment decisions
            """, unsafe_allow_html=True)
        
        st.success("""
        💡 **Recommended Workflow:**
        1. **Screen** with Logistic Regression (cast wide net)
        2. **Refine** with Random Forest (narrow candidates)
        3. **Validate** with XGBoost (final selection)
        4. **Expert Review** for high-value sites
        """)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Use Logistic Regression When:
            - 🔍 Conducting initial site screening
            - 🎯 Want to find ALL potential sites (high recall)
            - ⚡ Need fast predictions
            - 📊 Require interpretable linear relationships
            - 💰 Can tolerate more false positives
            """)
        
        with col2:
            st.markdown("""
            #### Use Random Forest When:
            - ✅ Performing final site validation
            - 🎯 Want to minimize false alarms (high precision)
            - 💰 False positives are costly
            - 📈 Complex non-linear relationships exist
            - 🔬 Need feature importance analysis
            """)
        
        st.info("💡 **Best Practice:** Use Logistic Regression for initial screening → Random Forest for final validation → Human expert review")

# TAB 4: About Project
with tab4:
    st.header("About This Project")
    
    st.markdown("""
    ### 🌬️ Wind Turbine Location Prediction
    
    This interactive demo showcases machine learning models developed by **Team CRANBerry** as part of the **AI4ALL Ignite** program. 
    Our project aims to predict optimal wind farm locations using AI-driven site selection.
    
    ---
    
    ### 👥 Team Members
    - **Christina** - Stony Brook University (Political Science, 3rd year)
    - **Ryan** - Morehouse College (CS, 2nd year)
    - **Aishwari** - UC Davis (CS, 2nd year)
    - **Nnenna** - Texas A&M University (Computer Engineering, 2nd year)
    - **Bemnet** - Texas State University (Math & CS, 2nd year)
    
    ---
    
    ### 🎯 Project Goals
    
    1. **Predict turbine-suitable locations** based on wind resource characteristics
    2. **Compare model performance** to understand tradeoffs between different approaches
    3. **Provide transparent, interpretable predictions** for both technical and public audiences
    4. **Demonstrate iterative ML workflow** from baseline to production model
    5. **Lay foundation** for future integration of environmental and social factors
    
    ---
    
    ### 📊 Data Sources
    
    - **USWTDB** (US Wind Turbine Database): 70,221 turbine locations with physical characteristics
    - **NREL Wind Toolkit**: 2.5M+ grid cells with wind resource data (2007-2013)
    - **Spatial Matching**: 25km radius tolerance for geospatial joins
    
    ---
    
    ### 🤖 Machine Learning Approach
    
    **Phase 1: Baseline Models**
    
    **Model 1: Logistic Regression**
    - Linear baseline model with balanced class weights
    - StandardScaler preprocessing
    - 4 base features
    - ROC-AUC: 0.732, Accuracy: 64.3%
    
    **Model 2: Random Forest**
    - Ensemble of 500 decision trees
    - Max 16 leaf nodes to prevent overfitting
    - 4 base features
    - ROC-AUC: 0.770, Accuracy: 70.3%
    
    **Model 3: Initial XGBoost**
    - Gradient boosting framework
    - 300 estimators, learning rate 0.05
    - 4 base features
    - ROC-AUC: 0.847, Accuracy: 76.6%
    
    **Phase 2: Feature Engineering & Optimization**
    
    **Model 4: Feature-Engineered XGBoost ⭐**
    - GridSearchCV hyperparameter tuning
    - **56 features** including:
      - Base: fraction_of_usable_area, capacity_factor
      - Numeric: wind_speed, capacity
      - Geographic: State (one-hot encoded, 50 columns)
      - Engineered: combined_wind_resource, potential_with_constraints
    - **Best Performance:**
      - ROC-AUC: **0.9545** (+12.7% over initial XGBoost)
      - Accuracy: **87.8%** (+11.2% improvement)
      - Turbine Recall: **94.3%** (finds nearly all turbine sites)
    
    ---
    
    ### 📈 Key Results
    
    **Model Evolution:**
    - **Baseline → Final:** +30.4% ROC-AUC improvement (0.732 → 0.9545)
    - **Feature Engineering:** Largest single improvement (+8.3% ROC-AUC)
    - **Hyperparameter Tuning:** Additional +2.4% ROC-AUC refinement
    
    **Best Model (XGBoost):**
    - Exceptional turbine detection (94.3% recall)
    - High precision for both classes (96.0% / 78.0%)
    - Production-ready performance
    - Deployed in this Streamlit app
    
    ---
    
    ### 🔮 Future Extensions
    
    1. Integrate environmental sensitivity data (wildlife corridors, bird migration)
    2. Add cost optimization (transmission infrastructure distance)
    3. Incorporate community acceptance factors
    4. Extend to hybrid renewable models (solar + wind)
    5. Deploy real-time API for industry partners
    6. Multi-temporal analysis for climate change impacts
    
    ---
    
    ### ⚖️ Responsible AI
    
    **Bias Mitigation:**
    - Balanced class weights to address data imbalance
    - Feature importance transparency
    - Multiple model comparison for robustness
    - Explicit documentation of limitations
    
    **Limitations:**
    - Does not include wildlife, noise, or community factors
    - Geographic bias toward accessible regions
    - Technology generation bias (older turbines over-represented)
    - Temporal measurement bias (2007-2013 data)
    - State-level encoding may not capture micro-climate variations
    
    **Ethical Considerations:**
    - Model predictions should augment, not replace, expert judgment
    - Environmental impact assessments still required
    - Community engagement essential for project success
    - Results should be validated with local stakeholder input
    
    ---
    
    ### 🔗 Links & Resources
    
    - [GitHub Repository](https://github.com/NnennaN123/AI4ALL-Project)
    - [Live Streamlit App](https://cranberryai4allproject.streamlit.app/)
    - [NREL Wind Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)
    - [USWTDB](https://eerscmap.usgs.gov/uswtdb/)
    
    ---
    
    ### 📧 Contact
    
    For questions about this project, please visit our GitHub repository or contact the team through AI4ALL.
    
    ---
    
    ### 🙏 Acknowledgments
    
    Special thanks to:
    - AI4ALL Ignite program mentors and organizers
    - National Renewable Energy Laboratory (NREL)
    - U.S. Geological Survey (USGS)
    - Lawrence Berkeley National Laboratory
    - Open source ML community (scikit-learn, XGBoost, Streamlit)
    
    ---
    
    ### 📚 Technical Documentation
    
    For detailed technical documentation, including:
    - Feature engineering process (12 combinations tested)
    - Hyperparameter tuning methodology
    - Model comparison analysis
    - Complete metrics and evaluation
    
    Visit our [GitHub Repository](https://github.com/NnennaN123/AI4ALL-Project)
    
    ---
    
    *Built with ❤️ by Team CRANBerry | AI4ALL 2024-2025*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Wind Turbine Site Predictor</strong> | Team CRANBerry | AI4ALL Ignite Program</p>
    <p>🔗 <a href='https://github.com/NnennaN123/AI4ALL-Project'>GitHub</a> | 
       📊 <a href='https://www.nrel.gov/'>NREL</a> | 
       🌍 <a href='https://eerscmap.usgs.gov/uswtdb/'>USWTDB</a></p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        ⭐ Latest Update: XGBoost model with feature engineering (ROC-AUC: 0.9545)
    </p>
</div>
""", unsafe_allow_html=True)