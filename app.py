import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hypertension Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Backgrounds and Styling
# Note: Replace the URLs below with your local file paths if you have local images
# e.g., url('images/main_bg.jpg')
main_bg_image = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80"
sidebar_bg_image = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60"

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("{main_bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("{sidebar_bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    background-color: rgba(0, 0, 0, 0.7); /* Dark overlay for sidebar */
}}

/* Text color adjustments for readability against backgrounds */
[data-testid="stAppViewContainer"] {{
    background-color: rgba(255, 255, 255, 0.9); /* White overlay for main content */
}}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    color: #2c3e50;
}}

.recommendation-box {{
    background-color: #e8f4f8;
    border-left: 5px solid #3498db;
    padding: 15px;
    margin-top: 20px;
    border-radius: 5px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD DATA & MODELS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load('xgboost_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        # Assuming you might have a logistic regression model saved, if not, this can be skipped
        # lr_model = joblib.load('logistic_regression_model.pkl') 
        return xgb_model, None # Return None for LR if not saved
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'xgboost_model.pkl' and 'random_forest_model.pkl' are in the directory.")
        return None, None

@st.cache_data
def load_feature_importance():
    try:
        df_imp = pd.read_csv('feature_importance_xgboost.csv')
        return df_imp
    except FileNotFoundError:
        return None

# Load models
xgb_model, lr_model = load_models()
feature_importance_df = load_feature_importance()

# Define feature names expected by the model (based on your notebook)
# Note: Ensure these match exactly what was used during training
feature_names = [
    'BMI', 'age', 'married', 'male.gender', 'hgb_centered', 
    'adv_HIV', 'arv_naive', 'urban.clinic', 'log_creat_centered'
]

# -----------------------------------------------------------------------------
# 3. SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.title("🩺 Patient Data Input")
st.sidebar.markdown("Please enter the patient's clinical details below.")

# Inputs
age = st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=45)
bmi = st.sidebar.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
male_gender_val = 1 if gender == "Male" else 0

married = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
married_val = 1 if married == "Married" else 0

urban_clinic = st.sidebar.selectbox("Clinic Location", ["Rural", "Urban"])
urban_val = 1 if urban_clinic == "Urban" else 0

# Clinical Metrics
sbp = st.sidebar.number_input("Systolic BP (mmHg)", min_value=80, max_value=250, value=120)
dbp = st.sidebar.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)

hgb = st.sidebar.number_input("Hemoglobin (g/dL) - Centered", value=0.0, format="%.2f", help="Enter centered Hgb value if known, else 0")
adv_hiv = st.sidebar.selectbox("Advanced HIV Status", ["No", "Yes"])
adv_hiv_val = 1 if adv_hiv == "Yes" else 0

arv_naive = st.sidebar.selectbox("ARV Naive (Not on treatment)", ["No (On ARV)", "Yes (Naive)"])
arv_naive_val = 1 if arv_naive == "Yes (Naive)" else 0

log_creat = st.sidebar.number_input("Log Creatinine (Centered)", value=0.0, format="%.4f", help="Enter centered Log Creatinine")

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD
# -----------------------------------------------------------------------------
st.title("🩺 Hypertension Risk Prediction Dashboard")
st.markdown("This tool uses Machine Learning to predict the likelihood of hypertension based on clinical and demographic factors.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Result")
    
    if xgb_model is not None:
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'BMI': [bmi],
            'age': [age],
            'married': [married_val],
            'male.gender': [male_gender_val],
            'hgb_centered': [hgb],
            'adv_HIV': [adv_hiv_val],
            'arv_naive': [arv_naive_val],
            'urban.clinic': [urban_val],
            'log_creat_centered': [log_creat]
        })
        
        # Ensure column order matches training
        input_data = input_data[feature_names]

        # Predict
        prediction_proba = xgb_model.predict_proba(input_data)[0][1]
        prediction_class = xgb_model.predict(input_data)[0]
        
        # Display Result
        if prediction_class == 1:
            st.error(f"**High Risk**: The model predicts a {prediction_proba*100:.2f}% probability of Hypertension.")
        else:
            st.success(f"**Low Risk**: The model predicts a {prediction_proba*100:.2f}% probability of Hypertension.")

        # --- CLINICAL RECOMMENDATIONS ENGINE ---
        st.markdown("---")
        st.subheader("📋 Clinical Recommendations")
        
        recommendations = []
        
        # BMI Logic
        if bmi >= 30:
            recommendations.append("🔴 **Obesity Alert**: BMI is ≥ 30. Recommend immediate lifestyle intervention (diet/exercise) and consider pharmacological support if comorbidities exist.")
        elif bmi >= 25:
            recommendations.append("🟡 **Overweight Alert**: BMI is ≥ 25. Recommend dietary counseling and increased physical activity.")
        
        # Age Logic
        if age > 50:
            recommendations.append("🟡 **Age Factor**: Patient is over 50. Regular monitoring of arterial stiffness and blood pressure is recommended.")
        
        # HIV Logic
        if adv_hiv_val == 1:
            recommendations.append("🔴 **Advanced HIV**: Increased risk of inflammation-induced hypertension. Monitor closely for drug interactions if starting antihypertensives.")
        
        if arv_naive_val == 1:
            recommendations.append("🟡 **ARV Naive**: Initiate ART consultation. Uncontrolled HIV contributes to vascular inflammation.")
        
        # Kidney Logic
        if log_creat > 0.5: # Arbitrary threshold for example, adjust based on clinical standards
            recommendations.append("🔴 **Kidney Function**: Elevated creatinine levels detected. Evaluate renal function immediately as it may be secondary to or causing hypertension.")

        # General BP Logic
        if sbp >= 140 or dbp >= 90:
            recommendations.append("🔴 **Current BP High**: Immediate confirmation of BP reading required. Consider initiating antihypertensive therapy per guidelines.")
        
        if not recommendations:
            recommendations.append("🟢 **Routine Care**: Continue routine monitoring. Maintain healthy lifestyle.")

        # Display Recommendations in a styled box
        rec_text = "\n\n".join(recommendations)
        st.markdown(f'<div class="recommendation-box">{rec_text}</div>', unsafe_allow_html=True)

    else:
        st.warning("Models are not loaded. Please check the file paths.")

with col2:
    st.subheader("Feature Importance")
    
    if feature_importance_df is not None:
        # Select Box for Model Selection (Even if we only have XGBoost data loaded, we can structure it for future)
        # For now, we use the loaded XGBoost data. If you have others, you can load them similarly.
        model_choice = st.selectbox("Select Model for Importance", ["XGBoost"]) 
        
        # Filter data based on selection (Currently only XGBoost is loaded from CSV)
        # If you add more CSVs, you can load them here and switch dataframes
        df_plot = feature_importance_df.copy()
        
        # Create Plotly Bar Chart
        fig = px.bar(
            df_plot, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title=f"Top Predictors ({model_choice})",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Feature importance calculated using Permutation Importance on the XGBoost model.")
    else:
        st.info("Feature importance data not available.")

# Footer
st.markdown("---")
st.markdown("© 2026 Hypertension Prediction System. For clinical decision support only.")
