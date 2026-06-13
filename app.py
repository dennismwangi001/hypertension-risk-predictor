import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hypertension AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Safe Model & Data Loading (Crash Prevention)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    required_models = ['xgboost', 'random_forest', 'knn']
    missing_files = []
    
    for m_name in required_models:
        path = f"{m_name}_model.pkl"
        if os.path.exists(path):
            models[m_name.replace('_', ' ').title()] = joblib.load(path)
        else:
            missing_files.append(path)
            
    if missing_files:
        st.error(f"❌ Missing model files: {', '.join(missing_files)}. Please run `train_models.py` first.")
        st.stop()
    return models

@st.cache_data
def load_metadata():
    if not os.path.exists('test_data.pkl') or not os.path.exists('feature_names.pkl'):
        st.error("❌ Missing metadata files. Please run `train_models.py` first.")
        st.stop()
    return joblib.load('test_data.pkl'), joblib.load('feature_names.pkl')

models = load_models()
test_data, feature_names = load_metadata()
X_test, y_test = test_data['X_test'], test_data['y_test']

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🧭 Navigation")
st.sidebar.image("https://via.placeholder.com/300x100/1f4e79/ffffff?text=Hypertension+AI", use_container_width=True)

page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🔮 Predict Risk", "📈 Model Performance", "🔍 Feature Importance", "ℹ️ About & Data Info"]
)

st.sidebar.markdown("---")
st.sidebar.success(f"✅ {len(models)} models loaded successfully")
st.sidebar.info("🛡️ Leakage columns (SBP, DBP, ID, SBP_ge120) were excluded during training.")

# -----------------------------------------------------------------------------
# 4. Page Routing
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("🩺 Hypertension Prediction Dashboard")
    st.markdown("Welcome to the Hypertension Risk Assessment Dashboard. This tool leverages advanced machine learning to predict hypertension risk while strictly avoiding data leakage.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Test Patients", len(X_test))
    col2.metric("Hypertension Cases", int(y_test.sum()))
    col3.metric("Normal Cases", int((y_test == 0).sum()))
    col4.metric("Active Models", len(models))
    
    st.markdown("---")
    st.subheader("📊 Target Variable Distribution (Test Set)")
    fig = px.pie(
        names=['Normal', 'Hypertension'], 
        values=[int((y_test == 0).sum()), int(y_test.sum())],
        color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
        hole=0.4
    )
    fig.update_layout(legend_title="Class")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predict Risk":
    st.title("🔮 Patient Risk Assessment")
    st.markdown("Enter the patient's clinical and demographic details below. The model will calculate the probability of hypertension.")
    
    with st.form("prediction_form"):
        st.markdown("### 📋 Patient Details")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age (years)", 18, 100, 40)
            bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 22.0, 0.1)
            hgb_centered = st.slider("Hemoglobin (Centered)", -10.0, 10.0, 0.0, 0.1)
            married = st.selectbox("Married", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            male_gender = st.selectbox("Male Gender", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            adv_HIV = st.selectbox("Advanced HIV", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            survtime = st.slider("Survival Time (days)", 0, 3000, 500, 10)
            event = st.selectbox("Event Occurred", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            arv_naive = st.selectbox("ARV Naive", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            urban_clinic = st.selectbox("Urban Clinic", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            log_creat_centered = st.slider("Log Creatinine (Centered)", -5.0, 5.0, 0.0, 0.1)
            
        submit_button = st.form_submit_button("🔍 Calculate Risk", type="primary", use_container_width=True)
        
    if submit_button:
        # CRASH PREVENTION: Ensure exact column order and names as training data
        input_data = pd.DataFrame({
            'age': [age], 'BMI': [bmi], 'married': [married], 'male.gender': [male_gender],
            'hgb_centered': [hgb_centered], 'adv_HIV': [adv_HIV], 'survtime': [survtime],
            'event': [event], 'arv_naive': [arv_naive], 'urban.clinic': [urban_clinic],
            'log_creat_centered': [log_creat_centered]
        })
        
        # Ensure columns are in the exact order the model expects
        input_data = input_data[feature_names]
        
        st.markdown("---")
        st.subheader("📊 Model Predictions")
        
        # Get predictions from all 3 models
        results = []
        for model_name, model in models.items():
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] * 100
            results.append({
                "Model": model_name,
                "Prediction": "High Risk ⚠️" if pred == 1 else "Low Risk ✅",
                "Probability": f"{prob:.1f}%"
            })
            
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        
        # Consensus logic
        high_risk_count = sum(1 for r in results if "High Risk" in r["Prediction"])
        if high_risk_count >= 2:
            st.error("⚠️ **Consensus: High Risk**\n\nThe majority of models predict a high risk of hypertension. Clinical evaluation is strongly recommended.")
        else:
            st.success("✅ **Consensus: Low Risk**\n\nThe majority of models predict a low risk. Continue routine health monitoring.")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Evaluation")
    st.markdown("Compare the predictive power of our top 3 models on the held-out test set.")
    
    model_name = st.selectbox("Select Model to Evaluate", list(models.keys()))
    model = models[model_name]
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
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
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal (0)', 'Hypertension (1)'],
            y=['Normal (0)', 'Hypertension (1)'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col_chart2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'ROC (AUC = {roc_auc_score(y_test, y_prob):.3f})', 
            line=dict(color='royalblue', width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', 
            name='Random Guess', line=dict(color='gray', width=2, dash='dash')
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.05]), height=400
        )
        st.plotly_chart(fig_roc, use_container_width=True)

elif page == "🔍 Feature Importance":
    st.title("🔍 Feature Importance")
    st.markdown("Understanding which features drive the model's predictions is crucial for clinical interpretability.")
    
    st.info("💡 *Note: KNN does not have native feature importance. Displaying XGBoost and Random Forest.*")
    
    tab1, tab2 = st.tabs(["🌳 Random Forest", "🚀 XGBoost"])
    
    with tab1:
        rf_model = models['Random Forest']
        # Extract feature names from the preprocessor
        preprocessor = rf_model.named_steps['preprocessor']
        # Get feature names after one-hot encoding
        try:
            feature_names_out = preprocessor.get_feature_names_out()
        except:
            feature_names_out = feature_names
            
        importances = rf_model.named_steps['classifier'].feature_importances_
        
        # Create a dataframe and sort
        feat_imp_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True).tail(10)
        
        fig = px.bar(
            feat_imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        xgb_model = models['XGBoost']
        preprocessor = xgb_model.named_steps['preprocessor']
        try:
            feature_names_out = preprocessor.get_feature_names_out()
        except:
            feature_names_out = feature_names
            
        importances = xgb_model.named_steps['classifier'].feature_importances_
        
        feat_imp_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True).tail(10)
        
        fig = px.bar(
            feat_imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Tealgrn'
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

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
    3. **K-Nearest Neighbors (KNN)**: A distance-based algorithm that finds similarities with past patients.
    
    ### ⚠️ Disclaimer
    This tool is for **educational and demonstrative purposes only**. It should not replace professional medical diagnosis or advice.
    """)