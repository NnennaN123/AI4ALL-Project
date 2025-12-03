"""
Wind Turbine Location Prediction - Interactive Demo
CRANBerry Team - AI4ALL Project

This Streamlit app allows users to interactively test our wind turbine prediction models.
Users can input site characteristics or select from real locations to see predictions.
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
        
        # Load metrics
        with open(BASE_DIR / 'model_metrics.json', 'r') as f:
            log_reg_metrics = json.load(f)
        
        with open(BASE_DIR / 'random_forest_model_metrics.json', 'r') as f:
            rf_metrics = json.load(f)
        
        return log_reg, rf_model, scaler, log_reg_metrics, rf_metrics
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        st.error(f"Looking in directory: {Path(__file__).parent}")
        st.stop()

# Load models
log_reg_model, rf_model, scaler, log_metrics, rf_metrics = load_models_and_metrics()

# Sample locations with real characteristics
SAMPLE_LOCATIONS = {
    "🌟 West Texas (High Potential)": {
        "latitude": 31.5,
        "longitude": -102.0,
        "fraction_of_usable_area": 0.85,
        "capacity": 2500,
        "wind_speed": 8.2,
        "capacity_factor": 0.42,
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

def create_feature_importance_chart():
    """Create feature importance visualization for Random Forest."""
    # Get feature importance from the model
    feature_names = ['Fraction Usable Area', 'Capacity', 'Wind Speed', 'Capacity Factor']
    importances = rf_model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure(go.Bar(
        x=[importances[i] * 100 for i in indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker=dict(
            color=[importances[i] for i in indices],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{importances[i]*100:.1f}%' for i in indices],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Random Forest Feature Importance',
        xaxis_title='Importance (%)',
        yaxis_title='Feature',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Title and description
st.markdown('<h1 class="main-header">🌬️ Wind Turbine Site Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicting Optimal Wind Farm Locations Using Machine Learning | Team CRANBerry - AI4ALL</p>', unsafe_allow_html=True)

# Sidebar - Model selection and information
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4a8.png", width=80)
    st.title("⚙️ Configuration")
    
    model_choice = st.radio(
        "Select Prediction Model:",
        ["🔵 Logistic Regression", "🟢 Random Forest", "🔄 Compare Both"],
        help="Choose which model to use for predictions"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LR ROC-AUC", f"{log_metrics['roc_auc']:.3f}")
        st.metric("LR Accuracy", f"{log_metrics['classification_report']['accuracy']:.1%}")
    
    with col2:
        st.metric("RF ROC-AUC", f"{rf_metrics['roc_auc']:.3f}")
        st.metric("RF Accuracy", f"{rf_metrics['classification_report']['accuracy']:.1%}")
    
    st.markdown("---")
    st.markdown("### 🎯 Input Features")
    st.markdown("""
    **1. Fraction of Usable Area** (0-1)  
    *Proportion of land suitable for development*
    
    **2. Capacity** (kW)  
    *Potential power generation*
    
    **3. Wind Speed** (m/s)  
    *Average wind speed at hub height*
    
    **4. Capacity Factor** (0-1)  
    *Expected efficiency ratio*
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Model Info")
    with st.expander("Logistic Regression"):
        st.markdown(f"""
        - **Recall:** {log_metrics['classification_report']['True']['recall']:.1%}
        - **Precision:** {log_metrics['classification_report']['True']['precision']:.1%}
        - **Best for:** Initial screening (high recall)
        - **Speed:** Very fast
        """)
    
    with st.expander("Random Forest"):
        st.markdown(f"""
        - **Recall:** {rf_metrics['classification_report']['True']['recall']:.1%}
        - **Precision:** {rf_metrics['classification_report']['True']['precision']:.1%}
        - **Best for:** Final validation (high precision)
        - **Speed:** Fast
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("""
    [GitHub Repository](https://github.com/NnennaN123/AI4ALL-Project)  
    [NREL Wind Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)  
    [USWTDB Database](https://eerscmap.usgs.gov/uswtdb/)
    """)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Interactive Predictor", "📍 Real Locations", "📈 Model Analysis", "ℹ️ About Project"])

# TAB 1: Interactive Predictor
with tab1:
    st.header("Test Custom Site Characteristics")
    st.markdown("Adjust the sliders below to see how different factors affect wind turbine suitability predictions.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎛️ Input Features")
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        fraction_usable = st.slider(
            "**Fraction of Usable Area**",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Proportion of the grid cell suitable for wind development (0 = none, 1 = all usable)"
        )
        st.caption("💡 Higher values indicate more available land for turbine placement")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        capacity = st.slider(
            "**Capacity (kW)**",
            min_value=16,
            max_value=5000,
            value=2000,
            step=100,
            help="Potential power generation capacity of the site"
        )
        st.caption("💡 Modern turbines typically range from 2000-3000 kW")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        wind_speed = st.slider(
            "**Average Wind Speed (m/s)**",
            min_value=4.0,
            max_value=12.0,
            value=7.5,
            step=0.1,
            help="Average wind speed at hub height (typically 80-100m)"
        )
        st.caption("💡 Commercial wind farms typically need >6.5 m/s")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        capacity_factor = st.slider(
            "**Capacity Factor**",
            min_value=0.1,
            max_value=0.6,
            value=0.35,
            step=0.01,
            help="Expected efficiency (energy produced / theoretical maximum)"
        )
        st.caption("💡 Good sites have capacity factors >0.30")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create feature array
        features = np.array([[fraction_usable, capacity, wind_speed, capacity_factor]])
        
        # Display feature summary
        st.markdown("---")
        st.markdown("### 📋 Current Configuration")
        feature_df = pd.DataFrame({
            'Feature': ['Usable Area', 'Capacity', 'Wind Speed', 'Capacity Factor'],
            'Value': [f'{fraction_usable:.2f}', f'{capacity} kW', f'{wind_speed:.1f} m/s', f'{capacity_factor:.2f}'],
            'Rating': [
                '🟢 Excellent' if fraction_usable > 0.8 else '🟡 Good' if fraction_usable > 0.6 else '🔴 Poor',
                '🟢 High' if capacity > 2500 else '🟡 Medium' if capacity > 1500 else '🔴 Low',
                '🟢 Strong' if wind_speed > 7.5 else '🟡 Moderate' if wind_speed > 6.5 else '🔴 Weak',
                '🟢 Excellent' if capacity_factor > 0.35 else '🟡 Good' if capacity_factor > 0.25 else '🔴 Poor'
            ]
        })
        st.dataframe(feature_df, hide_index=True, use_container_width=True)
        
    with col2:
        st.subheader("🎯 Predictions")
        
        # Make predictions based on model choice
        if model_choice == "🔵 Logistic Regression":
            pred, prob = make_prediction(features, log_reg_model, use_scaler=True)
            
            prediction_text = "✓ Turbine Likely Present" if pred else "✗ Turbine Unlikely"
            prediction_class = "turbine-yes" if pred else "turbine-no"
            
            st.markdown(f'<div class="prediction-box {prediction_class}">{prediction_text}</div>', unsafe_allow_html=True)
            
            confidence_level, confidence_class = get_confidence_level(prob[1])
            st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence_level}</span> ({prob[1]*100:.1f}% probability)", unsafe_allow_html=True)
            
            # Gauge chart
            st.plotly_chart(create_probability_gauge(prob[1], "Turbine Presence Probability"), use_container_width=True)
            
            # Probability breakdown
            st.markdown("---")
            st.markdown("### 📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['No Turbine', 'Turbine Present'],
                'Probability': [f'{prob[0]*100:.2f}%', f'{prob[1]*100:.2f}%']
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
            
        elif model_choice == "🟢 Random Forest":
            pred, prob = make_prediction(features, rf_model, use_scaler=False)
            
            prediction_text = "✓ Turbine Likely Present" if pred else "✗ Turbine Unlikely"
            prediction_class = "turbine-yes" if pred else "turbine-no"
            
            st.markdown(f'<div class="prediction-box {prediction_class}">{prediction_text}</div>', unsafe_allow_html=True)
            
            confidence_level, confidence_class = get_confidence_level(prob[1])
            st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence_level}</span> ({prob[1]*100:.1f}% probability)", unsafe_allow_html=True)
            
            # Gauge chart
            st.plotly_chart(create_probability_gauge(prob[1], "Turbine Presence Probability"), use_container_width=True)
            
            # Probability breakdown
            st.markdown("---")
            st.markdown("### 📊 Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': ['No Turbine', 'Turbine Present'],
                'Probability': [f'{prob[0]*100:.2f}%', f'{prob[1]*100:.2f}%']
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
            
        else:  # Compare Both
            pred_lr, prob_lr = make_prediction(features, log_reg_model, use_scaler=True)
            pred_rf, prob_rf = make_prediction(features, rf_model, use_scaler=False)
            
            st.markdown("#### 🔵 Logistic Regression")
            prediction_text_lr = "✓ Turbine Likely" if pred_lr else "✗ No Turbine"
            prediction_class_lr = "turbine-yes" if pred_lr else "turbine-no"
            st.markdown(f'<div class="prediction-box {prediction_class_lr}">{prediction_text_lr}</div>', unsafe_allow_html=True)
            
            confidence_level_lr, confidence_class_lr = get_confidence_level(prob_lr[1])
            st.markdown(f"**Confidence:** <span class='{confidence_class_lr}'>{confidence_level_lr}</span> ({prob_lr[1]*100:.1f}%)", unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("#### 🟢 Random Forest")
            prediction_text_rf = "✓ Turbine Likely" if pred_rf else "✗ No Turbine"
            prediction_class_rf = "turbine-yes" if pred_rf else "turbine-no"
            st.markdown(f'<div class="prediction-box {prediction_class_rf}">{prediction_text_rf}</div>', unsafe_allow_html=True)
            
            confidence_level_rf, confidence_class_rf = get_confidence_level(prob_rf[1])
            st.markdown(f"**Confidence:** <span class='{confidence_class_rf}'>{confidence_level_rf}</span> ({prob_rf[1]*100:.1f}%)", unsafe_allow_html=True)
            
            # Comparison chart
            st.markdown("---")
            st.markdown("### 📊 Model Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': ['Logistic Regression', 'Random Forest'],
                'Prediction': [prediction_text_lr, prediction_text_rf],
                'Probability': [f'{prob_lr[1]*100:.1f}%', f'{prob_rf[1]*100:.1f}%'],
                'Confidence': [confidence_level_lr, confidence_level_rf]
            })
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            # Side-by-side gauges
            gauge_col1, gauge_col2 = st.columns(2)
            with gauge_col1:
                st.plotly_chart(create_probability_gauge(prob_lr[1], "Logistic Regression"), use_container_width=True)
            with gauge_col2:
                st.plotly_chart(create_probability_gauge(prob_rf[1], "Random Forest"), use_container_width=True)

# TAB 2: Real Locations
with tab2:
    st.header("Test Real-World Locations")
    st.markdown("Select from actual locations across the United States to see how our models perform on real data.")
    
    # Location selector
    selected_location = st.selectbox(
        "Choose a location:",
        options=list(SAMPLE_LOCATIONS.keys()),
        help="These are real locations with actual wind characteristics"
    )
    
    location_data = SAMPLE_LOCATIONS[selected_location]
    
    # Display location information
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📍 Location Details")
        st.markdown(f"**Coordinates:** {location_data['latitude']}°N, {abs(location_data['longitude'])}°W")
        st.markdown(f"**Description:** {location_data['description']}")
        st.markdown(f"**Ground Truth:** {location_data['actual']}")
        
        st.markdown("---")
        st.markdown("### 📊 Site Characteristics")
        
        location_features = pd.DataFrame({
            'Feature': ['Fraction Usable Area', 'Capacity', 'Wind Speed', 'Capacity Factor'],
            'Value': [
                f"{location_data['fraction_of_usable_area']:.2f}",
                f"{location_data['capacity']} kW",
                f"{location_data['wind_speed']:.1f} m/s",
                f"{location_data['capacity_factor']:.2f}"
            ]
        })
        st.dataframe(location_features, hide_index=True, use_container_width=True)
        
        # Feature visualization
        fig_features = go.Figure()
        
        features_normalized = [
            location_data['fraction_of_usable_area'],
            location_data['capacity'] / 5000,
            location_data['wind_speed'] / 12.0,
            location_data['capacity_factor'] / 0.6
        ]
        
        fig_features.add_trace(go.Bar(
            x=['Usable Area', 'Capacity', 'Wind Speed', 'Cap. Factor'],
            y=[f*100 for f in features_normalized],
            marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'],
            text=[f'{f*100:.1f}%' for f in features_normalized],
            textposition='auto',
        ))
        
        fig_features.update_layout(
            title='Feature Values (Normalized to 100%)',
            yaxis_title='Percentage of Maximum',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_features, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Model Predictions")
        
        # Prepare features
        loc_features = np.array([[
            location_data['fraction_of_usable_area'],
            location_data['capacity'],
            location_data['wind_speed'],
            location_data['capacity_factor']
        ]])
        
        # Make predictions
        pred_lr, prob_lr = make_prediction(loc_features, log_reg_model, use_scaler=True)
        pred_rf, prob_rf = make_prediction(loc_features, rf_model, use_scaler=False)
        
        # Display predictions
        st.markdown("#### 🔵 Logistic Regression")
        prediction_text_lr = "✓ Turbine Likely Present" if pred_lr else "✗ Turbine Unlikely"
        prediction_class_lr = "turbine-yes" if pred_lr else "turbine-no"
        st.markdown(f'<div class="prediction-box {prediction_class_lr}">{prediction_text_lr}</div>', unsafe_allow_html=True)
        st.markdown(f"**Probability:** {prob_lr[1]*100:.1f}%")
        
        st.markdown("---")
        
        st.markdown("#### 🟢 Random Forest")
        prediction_text_rf = "✓ Turbine Likely Present" if pred_rf else "✗ Turbine Unlikely"
        prediction_class_rf = "turbine-yes" if pred_rf else "turbine-no"
        st.markdown(f'<div class="prediction-box {prediction_class_rf}">{prediction_text_rf}</div>', unsafe_allow_html=True)
        st.markdown(f"**Probability:** {prob_rf[1]*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Comparison")
        
        # Comparison bar chart
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Logistic Regression',
            x=['No Turbine', 'Turbine Present'],
            y=[prob_lr[0]*100, prob_lr[1]*100],
            marker_color='#667eea',
            text=[f'{prob_lr[0]*100:.1f}%', f'{prob_lr[1]*100:.1f}%'],
            textposition='auto',
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Random Forest',
            x=['No Turbine', 'Turbine Present'],
            y=[prob_rf[0]*100, prob_rf[1]*100],
            marker_color='#10B981',
            text=[f'{prob_rf[0]*100:.1f}%', f'{prob_rf[1]*100:.1f}%'],
            textposition='auto',
        ))
        
        fig_comparison.update_layout(
            title='Model Probability Comparison',
            yaxis_title='Probability (%)',
            barmode='group',
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Batch test all locations
    st.markdown("---")
    st.markdown("### 🗺️ All Locations Summary")
    
    if st.button("🚀 Test All Locations"):
        results = []
        
        for loc_name, loc_data in SAMPLE_LOCATIONS.items():
            loc_features = np.array([[
                loc_data['fraction_of_usable_area'],
                loc_data['capacity'],
                loc_data['wind_speed'],
                loc_data['capacity_factor']
            ]])
            
            pred_lr, prob_lr = make_prediction(loc_features, log_reg_model, use_scaler=True)
            pred_rf, prob_rf = make_prediction(loc_features, rf_model, use_scaler=False)
            
            results.append({
                'Location': loc_name,
                'Ground Truth': loc_data['actual'],
                'LR Prediction': 'Turbine' if pred_lr else 'No Turbine',
                'LR Probability': f"{prob_lr[1]*100:.1f}%",
                'RF Prediction': 'Turbine' if pred_rf else 'No Turbine',
                'RF Probability': f"{prob_rf[1]*100:.1f}%"
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, hide_index=True, use_container_width=True)

# TAB 3: Model Analysis
with tab3:
    st.header("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Logistic Regression Metrics")
        st.metric("ROC-AUC Score", f"{log_metrics['roc_auc']:.3f}")
        st.metric("Overall Accuracy", f"{log_metrics['classification_report']['accuracy']:.1%}")
        
        st.markdown("#### Classification Report")
        log_class_df = pd.DataFrame({
            'Class': ['No Turbine', 'Turbine Present'],
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
        
        st.markdown("**💡 Key Strength:** High recall (72.8%) - great for initial screening to find all potential sites")
        
    with col2:
        st.markdown("### 🟢 Random Forest Metrics")
        st.metric("ROC-AUC Score", f"{rf_metrics['roc_auc']:.3f}")
        st.metric("Overall Accuracy", f"{rf_metrics['classification_report']['accuracy']:.1%}")
        
        st.markdown("#### Classification Report")
        rf_class_df = pd.DataFrame({
            'Class': ['No Turbine', 'Turbine Present'],
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
    
    # Model comparison visualization
    st.markdown("---")
    st.markdown("### 📊 Model Performance Comparison")
    
    comparison_metrics = {
        'Metric': ['ROC-AUC', 'Accuracy', 'Precision (Turbine)', 'Recall (Turbine)', 'F1-Score (Turbine)'],
        'Logistic Regression': [
            log_metrics['roc_auc'],
            log_metrics['classification_report']['accuracy'],
            log_metrics['classification_report']['True']['precision'],
            log_metrics['classification_report']['True']['recall'],
            log_metrics['classification_report']['True']['f1-score']
        ],
        'Random Forest': [
            rf_metrics['roc_auc'],
            rf_metrics['classification_report']['accuracy'],
            rf_metrics['classification_report']['True']['precision'],
            rf_metrics['classification_report']['True']['recall'],
            rf_metrics['classification_report']['True']['f1-score']
        ]
    }
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Logistic Regression',
        x=comparison_metrics['Metric'],
        y=comparison_metrics['Logistic Regression'],
        marker_color='#667eea',
        text=[f'{v:.3f}' for v in comparison_metrics['Logistic Regression']],
        textposition='auto',
    ))
    
    fig_comparison.add_trace(go.Bar(
        name='Random Forest',
        x=comparison_metrics['Metric'],
        y=comparison_metrics['Random Forest'],
        marker_color='#10B981',
        text=[f'{v:.3f}' for v in comparison_metrics['Random Forest']],
        textposition='auto',
    ))
    
    fig_comparison.update_layout(
        title='Side-by-Side Metric Comparison',
        yaxis_title='Score',
        barmode='group',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Feature importance
    st.markdown("---")
    st.markdown("### 🎯 Feature Importance (Random Forest)")
    st.plotly_chart(create_feature_importance_chart(), use_container_width=True)
    
    st.markdown("""
    **Insights:**
    - **Capacity Factor** is the most important feature (~38%), indicating that efficiency is the primary driver
    - **Wind Speed** is second (~31%), confirming the importance of wind resources
    - **Capacity** and **Usable Area** play supporting roles in the prediction
    """)
    
    # Model recommendations
    st.markdown("---")
    st.markdown("### 🎯 Model Selection Guide")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("""
        #### Use Logistic Regression When:
        - 🔍 Conducting initial site screening
        - 🎯 Want to find ALL potential sites (high recall)
        - ⚡ Need fast predictions
        - 📊 Require interpretable linear relationships
        - 💰 Can tolerate more false positives
        """)
    
    with rec_col2:
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
    2. **Compare model performance** to understand tradeoffs between recall and precision
    3. **Provide transparent, interpretable predictions** for both technical and public audiences
    4. **Lay foundation** for future integration of environmental and social factors
    
    ---
    
    ### 📊 Data Sources
    
    - **USWTDB** (US Wind Turbine Database): 70,221 turbine locations with physical characteristics
    - **NREL Wind Toolkit**: 2.5M+ grid cells with wind resource data (2007-2013)
    - **Spatial Matching**: 25km radius tolerance for geospatial joins
    
    ---
    
    ### 🤖 Machine Learning Approach
    
    **Model 1: Logistic Regression**
    - Linear baseline model with balanced class weights
    - StandardScaler preprocessing
    - Optimized for high recall (72.8%)
    - ROC-AUC: 0.732
    
    **Model 2: Random Forest**
    - Ensemble of 500 decision trees
    - Max 16 leaf nodes to prevent overfitting
    - Optimized for precision (66.3%)
    - ROC-AUC: 0.770
    
    **Features Used:**
    1. Fraction of usable area (0-1)
    2. Capacity (kW)
    3. Wind speed (m/s)
    4. Capacity factor (0-1)
    
    ---
    
    ### 📈 Key Results
    
    - **Random Forest** outperforms Logistic Regression in overall accuracy (70.3% vs 64.3%)
    - **Logistic Regression** excels at finding all potential sites (recall: 72.8%)
    - **Random Forest** minimizes false positives (precision: 66.3%)
    - Both models significantly outperform random guessing
    
    ---
    
    ### 🔮 Future Extensions
    
    1. Integrate environmental sensitivity data (wildlife corridors, bird migration)
    2. Add cost optimization (transmission infrastructure distance)
    3. Incorporate community acceptance factors
    4. Extend to hybrid renewable models (solar + wind)
    5. Deploy real-time API for industry partners
    
    ---
    
    ### ⚖️ Responsible AI
    
    **Bias Mitigation:**
    - Balanced class weights to address data imbalance
    - Feature importance transparency
    - Dual reporting system (technical + public)
    - Explicit documentation of limitations
    
    **Limitations:**
    - Does not include wildlife, noise, or community factors
    - Geographic bias toward accessible regions
    - Technology generation bias (older turbines over-represented)
    - Temporal measurement bias
    
    ---
    
    ### 🔗 Links & Resources
    
    - [GitHub Repository](https://github.com/NnennaN123/AI4ALL-Project)
    - [NREL Wind Toolkit](https://www.nrel.gov/grid/wind-toolkit.html)
    - [USWTDB](https://eerscmap.usgs.gov/uswtdb/)
    - [Project Presentation](https://github.com/NnennaN123/AI4ALL-Project)
    
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
</div>
""", unsafe_allow_html=True)
