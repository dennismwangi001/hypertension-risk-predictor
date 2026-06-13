import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# -----------------------------------------------------------------------------
# 1. Custom CSS Styling for Readability & Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hypertension AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting custom CSS to fix contrast and add styling
st.markdown("""
<style>
    /* Global Background & Text Colors */
    .main {
        background-color: #0f172a; /* Deep Slate Blue */
        color: #e2e8f0; /* Light Gray Text */
    }
    
    /* Sidebar Styling - Glassmorphism Effect */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
        color: #f8fafc !important;
    }
    
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] p {
        color: #94a3b8 !important; /* Label Color */
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #38bdf8 !important; /* Value Color - Sky Blue */
        font-size: 2rem !important;
    }

    /* Headers & Titles */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Form Inputs Styling */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #475569;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0ea5e9;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0284c7;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
    }

    /* Alerts & Info Boxes */
    .stAlert {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Load Models & Data (Cached for speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models['XGBoost'] = joblib.load('xgboost_model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ xgboost_model.pkl not found.")
    try:
        models['Random Forest'] = joblib.load('random_forest_model.pkl')
    except FileNotFoundError:
        st.warning("⚠️ random_forest_model.pkl not found.")
    return models

@st.cache_data
def load_test_data():
    try:
        data = joblib.load('test_data.pkl')
        if isinstance(data, tuple):
            return data[0], data[1]
        elif isinstance(data, dict):
            return data.get('X_test'), data.get('y_test')
        else:
            return data, None
    except FileNotFoundError:
        st.error("❌ test_data.pkl not found.")
        return None, None

models = load_models()
X_test, y_test = load_test_data()

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")

# 🖼️ SIDEBAR IMAGE PLACEHOLDER
sidebar_image_url = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80"
st.sidebar.image(sidebar_image_url, use_container_width=True)
st.sidebar.caption("AI-Powered Health Diagnostics")

page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🔮 Predict Risk", "📈 Model Performance", "🔍 Feature Importance", "ℹ️ About & Data Info"]
)

st.sidebar.markdown("---")
if not models:
    st.sidebar.warning("⚠️ Models not loaded.")
else:
    st.sidebar.success(f"✅ {len(models)} model(s) active")

# -----------------------------------------------------------------------------
# 4. Pages
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("🩺 Hypertension Prediction Dashboard")
    
    # 🖼️ DASHBOARD BANNER IMAGE
    dashboard_image_url = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80"
    st.image(dashboard_image_url, use_container_width=True, caption="Advanced Clinical Decision Support System")
    
    st.markdown("### 📌 Overview")
    st.markdown("This dashboard provides real-time hypertension risk assessment using ensemble machine learning. All predictions are generated without relying on blood pressure metrics to ensure clinical applicability in pre-screening scenarios.")
    
    if X_test is not None and y_test is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Patients", len(X_test))
        col2.metric("Hypertension Cases", int(y_test.sum()))
        col3.metric("Normal Cases", int((y_test == 0).sum()))
        col4.metric("Active Models", len(models))
    else:
        st.warning("📂 Test data not loaded.")

    st.markdown("---")
    st.markdown("### 📊 Target Variable Distribution")
    if X_test is not None and y_test is not None:
        fig = px.pie(
            names=['Normal', 'Hypertension'], 
            values=[int((y_test == 0).sum()), int(y_test.sum())],
            color_discrete_sequence=['#10b981', '#ef4444'], # Emerald Green & Red
            hole=0.4
        )
        fig.update_layout(
            legend_title="Class",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predict Risk":
    st.title("🔮 Patient Risk Assessment")
    st.markdown("Enter the patient's clinical parameters below. The system will output a probability score and clinical recommendation.")
    
    with st.form("prediction_form"):
        st.markdown("### 📋 Patient Details")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age (years)", 18, 100, 40)
            bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 22.0, 0.1)
            hgb_centered = st.number_input("Hemoglobin (Centered)", value=0.0, step=0.1)
            married = st.selectbox("Married Status", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            male_gender = st.selectbox("Male Gender", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            adv_HIV = st.selectbox("Advanced HIV", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
        with col2:
            survtime = st.number_input("Survival Time (days)", 0, 5000, 500, step=10)
            event = st.selectbox("Event Occurred", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            arv_naive = st.selectbox("ARV Naive", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            urban_clinic = st.selectbox("Urban Clinic", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            log_creat_centered = st.number_input("Log Creatinine (Centered)", value=0.0, step=0.1)
            
        submit_button = st.form_submit_button("🔍 Calculate Risk", type="primary", use_container_width=True)
        
    if submit_button:
        if not models:
            st.error("No models loaded.")
        else:
            model = models.get('XGBoost') or list(models.values())[0]
            
            input_data = pd.DataFrame({
                'age': [age], 'BMI': [bmi], 'married': [married], 'male.gender': [male_gender],
                'hgb_centered': [hgb_centered], 'adv_HIV': [adv_HIV], 'survtime': [survtime],
                'event': [event], 'arv_naive': [arv_naive], 'urban.clinic': [urban_clinic],
                'log_creat_centered': [log_creat_centered]
            })
            
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] * 100
                
                st.markdown("---")
                if prediction == 1:
                    st.error(f"⚠️ **High Risk**: The model predicts a **{probability:.1f}%** chance of hypertension.")
                    st.markdown("💡 *Recommendation*: Schedule immediate clinical evaluation. Monitor BP, lifestyle factors, and consider pharmacological intervention per guidelines.")
                else:
                    st.success(f"✅ **Low Risk**: The model predicts a **{probability:.1f}%** chance of hypertension.")
                    st.markdown("💡 *Recommendation*: Continue routine health monitoring. Maintain healthy diet, exercise, and regular check-ups.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Evaluation")
    st.markdown("Compare predictive power across trained models on the held-out test set.")
    
    if not models:
        st.error("No models found.")
    else:
        model_name = st.selectbox("Select Model to Evaluate", list(models.keys()))
        model = models[model_name]
        
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            st.markdown("### 🎯 Key Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
            col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Normal (0)', 'Hypertension (1)'],
                                   y=['Normal (0)', 'Hypertension (1)'],
                                   color_continuous_scale='Blues')
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col_chart2:
                st.subheader("ROC Curve")
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc_score(y_test, y_prob):.3f})', line=dict(color='#38bdf8', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='#94a3b8', width=2, dash='dash')))
                fig_roc.update_layout(
                    xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', 
                    xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]), height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig_roc, use_container_width=True)

elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance Analysis")
    st.markdown("Understanding which features drive the model's predictions is crucial for clinical interpretability.")
    
    if 'XGBoost' in models:
        xgb_model = models['XGBoost']
        preprocessor = xgb_model.named_steps['preprocessor']
        feature_names_out = preprocessor.get_feature_names_out()
        importances = xgb_model.named_steps['classifier'].feature_importances_
        
        feat_imp_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True).tail(10)
        
        fig = px.bar(
            feat_imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=500, 
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("XGBoost model not found.")

elif page == "ℹ️ About & Data Info":
    st.title("ℹ️ About the Model & Data")
    
    st.markdown("""
    ### 🎯 Objective
    This application predicts the likelihood of hypertension based on clinical and demographic features, **strictly avoiding data leakage**.
    
    ### 🛡️ Data Leakage Prevention
    The following columns were explicitly **dropped** from the training data because they directly reveal or are derived from the target variable:
    - `ID` (Administrative)
    - `SBP` (Systolic Blood Pressure - Direct indicator)
    - `DBP` (Diastolic Blood Pressure - Direct indicator)
    - `SBP_ge120` (Derived directly from SBP)
    - `IPW_weight` (Administrative weighting)
    
    ### 🧠 Models Used
    1. **XGBoost**: A powerful gradient boosting algorithm, excellent for tabular data.
    2. **Random Forest**: An ensemble of decision trees, robust to overfitting.
    
    ### ⚠️ Disclaimer
    This tool is for **educational and demonstrative purposes only**. It should not replace professional medical diagnosis or advice.
    """)
