import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS with Backgrounds + Overlays
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hypertension AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define background images (replace URLs with your own if needed)
SIDEBAR_BG = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80"
MAIN_BG = "data:image/webp;base64,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"
# Inject CSS for backgrounds with overlays and readable text
st.markdown(f"""
<style>
    /* Main background with dark overlay */
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("{MAIN_BG}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Dark overlay for main content area */
    [data-testid="stAppViewContainer"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {{
        background-color: rgba(15, 23, 42, 0.85); /* Dark slate with transparency */
        min-height: 100vh;
        border-radius: 0;
    }}

    /* Sidebar background with dark overlay */
    [data-testid="stSidebar"] {{
        background-image: url("{SIDEBAR_BG}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(30, 41, 59, 0.92); /* Darker sidebar overlay */
        min-height: 100vh;
        backdrop-filter: blur(5px);
    }}

    /* Global text color adjustments for readability */
    .main h1, .main h2, .main h3, .main p, .main label, .main span {{
        color: #f1f5f9 !important; /* Light gray/white */
    }}
    
    /* Metric cards styling */
    div[data-testid="metric-container"] {{
        background-color: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }}
    div[data-testid="stMetricValue"] {{ 
        color: #38bdf8 !important; /* Sky blue */
        font-size: 1.8rem !important; 
    }}
    div[data-testid="stMetricLabel"] {{ 
        color: #94a3b8 !important; /* Light gray */
    }}

    /* Form inputs styling */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div {{
        background-color: rgba(30, 41, 59, 0.9);
        color: #f8fafc;
        border: 1px solid #475569;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: #0ea5e9;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: #0284c7;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
    }}

    /* Prediction result box */
    .prediction-box {{
        background-color: rgba(30, 41, 59, 0.9);
        border-left: 5px solid #0ea5e9;
        padding: 20px;
        margin-top: 20px;
        border-radius: 8px;
        color: #e2e8f0;
    }}
    
    /* Plotly charts background */
    .js-plotly-plot .plotly {{
        background-color: rgba(0,0,0,0) !important;
    }}
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
        if isinstance(data, dict):
            return data.get('X_test'), data.get('y_test')
        elif isinstance(data, tuple):
            return data[0], data[1]
        else:
            return data, None
    except FileNotFoundError:
        st.error("❌ test_data.pkl not found.")
        return None, None

models = load_models()
X_test, y_test = load_test_data()

# Exact feature order from your training script
FEATURE_ORDER = [
    'age', 'BMI', 'married', 'male.gender', 
    'hgb_centered', 'adv_HIV', 'survtime', 'event', 
    'arv_naive', 'urban.clinic', 'log_creat_centered'
]

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
st.sidebar.image(SIDEBAR_BG, use_container_width=True, caption="AI-Powered Health Diagnostics")

page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🔮 Predict Risk", "📈 Model Performance", "🔍 Feature Importance", "ℹ️ About & Data Info"]
)

st.sidebar.markdown("---")
if models:
    st.sidebar.success(f"✅ {len(models)} model(s) active")
else:
    st.sidebar.warning("️ No models loaded")

# -----------------------------------------------------------------------------
# 4. Pages
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("🩺 Hypertension Prediction Dashboard")
    st.image(MAIN_BG, use_container_width=True, caption="Advanced Clinical Decision Support System")
    
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
            color_discrete_sequence=['#10b981', '#ef4444'],
            hole=0.4
        )
        fig.update_layout(
            legend_title="Class",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == " Predict Risk":
    st.title("🔮 Patient Risk Assessment")
    st.markdown("Enter the patient's clinical parameters below. The system will output a probability score and clinical recommendation.")
    
    with st.form("prediction_form"):
        st.markdown("###  Patient Details")
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
            # Use XGBoost as primary model
            model = models.get('XGBoost') or list(models.values())[0]
            
            # Create DataFrame with EXACT feature names and order
            input_data = pd.DataFrame({
                'age': [age], 
                'BMI': [bmi], 
                'married': [married], 
                'male.gender': [male_gender],
                'hgb_centered': [hgb_centered], 
                'adv_HIV': [adv_HIV], 
                'survtime': [survtime],
                'event': [event], 
                'arv_naive': [arv_naive], 
                'urban.clinic': [urban_clinic],
                'log_creat_centered': [log_creat_centered]
            })
            
            # Ensure column order matches training
            input_data = input_data[FEATURE_ORDER]

            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1] * 100
                
                st.markdown("---")
                
                # Display result in a styled box
                if prediction == 1:
                    st.markdown(f'<div class="prediction-box"><h3>⚠️ High Risk</h3><p>The model predicts a <strong>{probability:.1f}%</strong> chance of hypertension.</p><p><em>Recommendation:</em> Schedule immediate clinical evaluation. Monitor BP, lifestyle factors, and consider pharmacological intervention per guidelines.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box"><h3>✅ Low Risk</h3><p>The model predicts a <strong>{probability:.1f}%</strong> chance of hypertension.</p><p><em>Recommendation:</em> Continue routine health monitoring. Maintain healthy diet, exercise, and regular check-ups.</p></div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Please ensure the model was trained with these exact features: " + ", ".join(FEATURE_ORDER))

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
            
            st.markdown("###  Key Metrics")
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
    
    ### ️ Disclaimer
    This tool is for **educational and demonstrative purposes only**. It should not replace professional medical diagnosis or advice.
    """)
