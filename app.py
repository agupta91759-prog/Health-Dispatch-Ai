import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. Page Config
st.set_page_config(page_title="Mobile Health Dispatch AI", layout="wide")

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# --- BUSINESS METRICS (INR) ---
COST_OF_NO_SHOW_INR = 800  
COST_OF_CALL_INR = 20      

# UI HEADER
st.title("🚑 Mobile Health Dispatch AI")
st.markdown("Predict the likelihood of a patient no-show to optimize mobile phlebotomy routing and reduce wasted drive time.")
st.markdown("---")

# Create Tabs for the Dashboard
tab1, tab2 = st.tabs(["📊 Single Patient Deep-Dive", "📅 Weekly Dispatch Command Center"])

# ==========================================
# TAB 1: SINGLE PATIENT CALCULATOR
# ==========================================
with tab1:
    st.sidebar.header("Single Patient Details (Deep Dive)")
    age = st.sidebar.slider("Patient Age", 0, 100, 35, key='age')
    lead_time = st.sidebar.slider("Lead Time (Days since booking)", 0, 60, 7, key='lead')
    drive_time = st.sidebar.slider("Drive Time to Patient (Mins)", 10, 60, 30, key='drive')

    st.sidebar.markdown("### Medical/History Flags")
    sms_received = st.sidebar.selectbox("Did patient receive SMS?", [0, 1], key='sms')
    scholarship = st.sidebar.selectbox("On Financial Aid?", [0, 1], key='schol')
    hipertension = st.sidebar.selectbox("Hypertension?", [0, 1], key='hyper')
    diabetes = st.sidebar.selectbox("Diabetes?", [0, 1], key='diab')
    alcoholism = st.sidebar.selectbox("Alcoholism?", [0, 1], key='alco')
    handcap = st.sidebar.selectbox("Handicap Level?", [0, 1, 2, 3, 4], index=0, key='hand')

    if st.button("Predict No-Show Risk", type="primary"):
        features = pd.DataFrame({
            'Age': [age], 'Lead_Time_Days': [lead_time], 'Drive_Time_Mins': [drive_time],
            'Scholarship': [scholarship], 'Hipertension': [hipertension], 'Diabetes': [diabetes],
            'Alcoholism': [alcoholism], 'Handcap': [handcap], 'SMS_received': [sms_received]
        })
        
        prob = model.predict_proba(features)[0][1]
        prob_percent = round(prob * 100, 1)
        expected_loss = round(prob * COST_OF_NO_SHOW_INR, 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Results")
            st.metric(label="Probability of Patient No-Show", value=f"{prob_percent}%")
            st.markdown("### Operations Recommendation")
            if expected_loss < COST_OF_CALL_INR:
                st.success("🟢 **LOW RISK:** Send standard automated SMS reminder.")
            elif prob_percent <= 50:
                st.warning("🟡 **MEDIUM RISK:** Require SMS confirmation 2 hours prior.")
            else:
                st.error("🔴 **HIGH RISK:** Mandatory human phone call required.")
                
        with col2:
            st.subheader("Financial Unit Economics")
            st.markdown(f"**Cost of Wasted Dispatch:** ₹{COST_OF_NO_SHOW_INR}")
            st.markdown(f"**Cost to Call Patient:** ₹{COST_OF_CALL_INR}")
            st.metric(label="Expected Financial Loss if ignored", value=f"₹{expected_loss}")
        
        st.markdown("---")
        st.subheader("Explainable AI: Why did the model make this decision?")
        
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(features)
        shap_vals_array = np.array(shap_vals)
        
        if isinstance(shap_vals, list):
            shap_vals_class1 = shap_vals[1][0]
        elif len(shap_vals_array.shape) == 3:
            shap_vals_class1 = shap_vals_array[0, :, 1]
        else:
            shap_vals_class1 = shap_vals_array[0]
            
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#ff4d4d' if val > 0 else '#4d94ff' for val in shap_vals_class1]
        ax.barh(features.columns, shap_vals_class1, color=colors)
        ax.set_xlabel("Impact on Risk Score")
        ax.set_title("Patient-Specific Feature Impact")
        st.pyplot(fig)


# ==========================================
# TAB 2: WEEKLY DISPATCH ROSTER (NEW)
# ==========================================
with tab2:
    st.subheader("📅 Weekly Dispatch Roster (AI Batch Processing)")
    st.markdown("Upload a week's worth of appointments (approx. 70-80 patients). The AI will instantly score the entire list and flag who needs manual intervention.")
    
    # 1. Generate Fake Weekly Data (10-12 patients a day for 7 days)
    np.random.seed(101)
    num_patients = 75 
    
    mock_data = pd.DataFrame({
        'Patient_ID': [f"PT-{np.random.randint(10000,99999)}" for _ in range(num_patients)],
        'Day': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], num_patients),
        'Age': np.random.randint(18, 85, num_patients),
        'Lead_Time_Days': np.random.randint(1, 45, num_patients),
        'Drive_Time_Mins': np.random.randint(10, 60, num_patients),
        'Scholarship': np.random.choice([0, 1], num_patients, p=[0.8, 0.2]),
        'Hipertension': np.random.choice([0, 1], num_patients, p=[0.7, 0.3]),
        'Diabetes': np.random.choice([0, 1], num_patients, p=[0.8, 0.2]),
        'Alcoholism': np.random.choice([0, 1], num_patients, p=[0.95, 0.05]),
        'Handcap': np.random.choice([0, 1, 2], num_patients, p=[0.9, 0.08, 0.02]),
        'SMS_received': np.random.choice([0, 1], num_patients, p=[0.5, 0.5])
    })
    
    # 2. Run the AI model on ALL 75 patients at once
    features_batch = mock_data[['Age', 'Lead_Time_Days', 'Drive_Time_Mins', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
    batch_probs = model.predict_proba(features_batch)[:, 1]
    
    # 3. Add AI scores to the table
    mock_data['Risk_Score_%'] = np.round(batch_probs * 100, 1)
    mock_data['Expected_Loss_₹'] = np.round(batch_probs * COST_OF_NO_SHOW_INR, 2)
    
    # 4. Assign Business Action Tags
    conditions = [
        (mock_data['Expected_Loss_₹'] > COST_OF_CALL_INR) & (mock_data['Risk_Score_%'] > 55),
        (mock_data['Expected_Loss_₹'] > COST_OF_CALL_INR) & (mock_data['Risk_Score_%'] <= 55)
    ]
    choices = ['🔴 CALL REQ', '🟡 SEND SMS']
    mock_data['AI_Recommendation'] = np.select(conditions, choices, default='🟢 AUTO')
    
    # 5. Display the DataFrame
    # Reorder columns to look clean
    display_cols = ['Day', 'Patient_ID', 'Drive_Time_Mins', 'Risk_Score_%', 'Expected_Loss_₹', 'AI_Recommendation']
    st.dataframe(mock_data[display_cols].sort_values(by=['Risk_Score_%'], ascending=False), height=400, use_container_width=True)
    
    # 6. Bulk Action Buttons
    st.markdown("---")
    st.subheader("⚡ Bulk Automated Actions")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📱 Bulk-Send SMS to all '🟡 SEND SMS' Patients", type="primary"):
            num_sms = len(mock_data[mock_data['AI_Recommendation'] == '🟡 SEND SMS'])
            with st.spinner(f"Sending {num_sms} messages via Twilio API..."):
                time.sleep(2)
            st.toast(f'Successfully delivered {num_sms} SMS reminders!', icon='✅')
            st.success(f"Automated System sent {num_sms} texts in 2.1 seconds.")
            
    with col_btn2:
        st.info("💡 **Dispatcher Note:** There are **{}** patients flagged as '🔴 CALL REQ'. Please call them manually before dispatching technicians.".format(len(mock_data[mock_data['AI_Recommendation'] == '🔴 CALL REQ'])))