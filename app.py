import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Backgrounds
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hypertension AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Image URLs (Replace with your own links if desired)
SIDEBAR_BG = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80"
MAIN_BG = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

# Inject CSS for Background Images and Readability
st.markdown(f"""
<style>
    /* Main Background Image with Overlay */
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("{MAIN_BG}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Semi-transparent overlay to ensure text readability on main bg */
    [data-testid="stAppViewContainer"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {{
        background-color: rgba(15, 23, 42, 0.85); 
        min-height: 100vh;
        border-radius: 0;
    }}

    /* Sidebar Background Image with Dark Overlay */
    [data-testid="stSidebar"] {{
        background-image: url("{SIDEBAR_BG}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(30, 41, 59, 0.92);
        min-height: 100vh;
        backdrop-filter: blur(5px);
    }}

    /* Global Text & Metric Styling */
    .main h1, .main h2, .main h3, .main p, .main label {{
        color: #f1f5f9 !important;
    }}
    div[data-testid="metric-container"] {{
        background-color: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }}
    div[data-testid="stMetricValue"] {{ color: #38bdf8 !important; font-size: 1.8rem !important; }}
    div[data-testid="stMetricLabel"] {{ color: #94a3b8 !important; }}

    /* Form Inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {{
        background-color: rgba(30, 41, 59, 0.9);
        color: #f8fafc;
        border: 1px solid #475569;
    }}
    
    /* Clinical Recommendation Box */
    .rec-box {{
        background-color: rgba(30, 41, 59, 0.9);
        border-left: 5px solid #0ea5e9;
        padding: 20px;
        margin-top: 20px;
        border-radius: 8px;
        color: #e2e8f0;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Load Models & Data
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try: models['XGBoost'] = joblib.load('xgboost_model.pkl')
    except FileNotFoundError: st.warning("⚠️ xgboost_model.pkl missing")
    try: models['Random Forest'] = joblib.load('random_forest_model.pkl')
    except FileNotFoundError: st.warning("⚠️ random_forest_model.pkl missing")
    return models

@st.cache_data
def load_test_data():
    try:
        data = joblib.load('test_data.pkl')
        if isinstance(data, dict): return data.get('X_test'), data.get('y_test')
        elif isinstance(data, tuple): return data[0], data[1]
        return data, None
    except Exception: return None, None

models = load_models()
X_test, y_test = load_test_data()

# Expected feature order from training
FEATURE_ORDER = ['age', 'BMI', 'married', 'male.gender', 'hgb_centered', 'adv_HIV', 'survtime', 'event', 'arv_naive', 'urban.clinic', 'log_creat_centered']

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
st.sidebar.image(SIDEBAR_BG, use_container_width=True, caption="AI-Powered Health Diagnostics")

page = st.sidebar.radio("Select Page", ["📊 Dashboard", " Predict Risk", "📈 Model Performance", "🔍 Feature Importance", "ℹ️ About & Data Info"])

st.sidebar.markdown("---")
if models: st.sidebar.success(f"✅ {len(models)} models active")
else: st.sidebar.warning("⚠️ No models loaded")

# -----------------------------------------------------------------------------
# 4. Pages
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("🩺 Hypertension Prediction Dashboard")
    st.image(MAIN_BG, use_container_width=True, caption="Advanced Clinical Decision Support System")
    
    st.markdown("### 📌 Overview")
    st.markdown("Real-time hypertension risk assessment using ensemble ML. Predictions exclude direct BP metrics to ensure clinical applicability.")
    
    if X_test is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Patients", len(X_test))
        c2.metric("Hypertension Cases", int(y_test.sum()))
        c3.metric("Normal Cases", int((y_test == 0).sum()))
        c4.metric("Active Models", len(models))
        
        st.markdown("---")
        fig = px.pie(names=['Normal', 'Hypertension'], values=[int((y_test==0).sum()), int(y_test.sum())], hole=0.4, color_discrete_sequence=['#10b981', '#ef4444'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predict Risk":
    st.title(" Patient Risk Assessment")
    
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 18, 100, 40)
            bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 22.0, 0.1)
            hgb = st.number_input("Hemoglobin (Centered)", value=0.0, step=0.1)
            married = st.selectbox("Married", [0,1], format_func=lambda x: "Yes" if x else "No")
            male = st.selectbox("Male Gender", [0,1], format_func=lambda x: "Yes" if x else "No")
            hiv = st.selectbox("Advanced HIV", [0,1], format_func=lambda x: "Yes" if x else "No")
        with c2:
            surv = st.number_input("Survival Time (days)", 0, 5000, 500, step=10)
            event = st.selectbox("Event Occurred", [0,1], format_func=lambda x: "Yes" if x else "No")
            arv = st.selectbox("ARV Naive", [0,1], format_func=lambda x: "Yes" if x else "No")
            urban = st.selectbox("Urban Clinic", [0,1], format_func=lambda x: "Yes" if x else "No")
            creat = st.number_input("Log Creatinine (Centered)", value=0.0, step=0.1)
            
        btn = st.form_submit_button(" Calculate Risk", type="primary", use_container_width=True)
        
    if btn and models:
        inp = pd.DataFrame({'age':[age], 'BMI':[bmi], 'married':[married], 'male.gender':[male], 'hgb_centered':[hgb], 'adv_HIV':[hiv], 'survtime':[surv], 'event':[event], 'arv_naive':[arv], 'urban.clinic':[urban], 'log_creat_centered':[creat]})
        inp = inp[FEATURE_ORDER]
        
        mdl = models.get('XGBoost') or list(models.values())[0]
        prob = mdl.predict_proba(inp)[0][1] * 100
        pred = mdl.predict(inp)[0]
        
        st.markdown("---")
        if pred == 1:
            st.error(f"️ **High Risk**: {prob:.1f}% probability of hypertension.")
        else:
            st.success(f"✅ **Low Risk**: {prob:.1f}% probability of hypertension.")
            
        # --- CLINICAL RECOMMENDATIONS ENGINE ---
        recs = []
        if bmi >= 30: recs.append("🔴 **Obesity Alert**: BMI ≥ 30. Recommend immediate lifestyle intervention (diet/exercise) and consider pharmacological support.")
        elif bmi >= 25: recs.append("🟡 **Overweight**: BMI ≥ 25. Recommend dietary counseling and increased physical activity.")
        if age > 50: recs.append("🟡 **Age Factor**: Patient > 50 years. Regular monitoring of arterial stiffness recommended.")
        if hiv == 1: recs.append("🔴 **Advanced HIV**: Monitor closely for inflammation-induced hypertension and drug interactions.")
        if arv == 1: recs.append("🟡 **ARV Naive**: Initiate ART consultation; uncontrolled HIV contributes to vascular inflammation.")
        if creat > 0.5: recs.append("🔴 **Kidney Function**: Elevated creatinine detected. Evaluate renal function immediately.")
        if not recs: recs.append("🟢 **Routine Care**: Continue routine monitoring and maintain healthy lifestyle.")
        
        st.markdown('<div class="rec-box"><h4>📋 Healthcare Provider Recommendations:</h4>' + '<br>'.join(recs) + '</div>', unsafe_allow_html=True)

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Evaluation")
    if models and X_test is not None:
        m_name = st.selectbox("Select Model", list(models.keys()))
        mdl = models[m_name]
        yp = mdl.predict(X_test); ypr = mdl.predict_proba(X_test)[:,1]
        
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy_score(y_test, yp):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, yp):.4f}")
        c3.metric("Recall", f"{recall_score(y_test, yp):.4f}")
        c4.metric("F1-Score", f"{f1_score(y_test, yp):.4f}")
        c5.metric("ROC-AUC", f"{roc_auc_score(y_test, ypr):.4f}")
        
        cc1, cc2 = st.columns(2)
        with cc1:
            cm = confusion_matrix(y_test, yp)
            fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="Actual"), x=['Normal','HTN'], y=['Normal','HTN'], color_continuous_scale='Blues')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            fpr, tpr, _ = roc_curve(y_test, ypr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc_score(y_test,ypr):.3f})', line=dict(color='#38bdf8', width=3)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(color='#94a3b8', dash='dash')))
            fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), height=400)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance Analysis")
    st.markdown("Select a model to view its specific feature drivers.")
    
    # ✅ DYNAMIC MODEL SELECTION BOX
    selected_model_name = st.selectbox("Choose Model for Importance View", list(models.keys()), key="fi_selector")
    
    if selected_model_name in models:
        mdl = models[selected_model_name]
        preproc = mdl.named_steps['preprocessor']
        feat_names = preproc.get_feature_names_out()
        imps = mdl.named_steps['classifier'].feature_importances_
        
        df_imp = pd.DataFrame({'Feature': feat_names, 'Importance': imps}).sort_values('Importance', ascending=True).tail(10)
        
        fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), height=500)
        st.plotly_chart(fig, use_container_width=True)

elif page == "ℹ️ About & Data Info":
    st.title("ℹ️ About the Model & Data")
    st.markdown("""
    ### 🎯 Objective
    Predicts hypertension likelihood while **strictly avoiding data leakage**.
    
    ### 🛡️ Leakage Prevention
    Excluded columns: `ID`, `SBP`, `DBP`, `SBP_ge120`, `IPW_weight`.
    
    ### ⚠️ Disclaimer
    For educational purposes only. Not a replacement for professional medical diagnosis.
    """)
