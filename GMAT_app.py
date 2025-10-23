import streamlit as st
import joblib
import numpy as np
from xgboost import XGBRegressor

# PAGE CONFIG
st.set_page_config(
    page_title="GMAT Score Predictor",
    page_icon="📊",
    layout="wide"
)

# LOAD MODEL + ENCODERS
@st.cache_resource
def load_models():
    xgb_model = joblib.load("xgb_model_gmat.pkl")
    le_board = joblib.load("le_board_gmat.pkl")
    le_gender = joblib.load("le_gender_gmat.pkl")
    scaler = joblib.load("scaler_gmat.pkl")
    return xgb_model, le_board, le_gender, scaler

xgb_model, le_board, le_gender, scaler = load_models()

# HEADER
st.title("📊 GMAT Score Predictor")
st.markdown("### Predict your GMAT score based on your academic and quantitative performance")
st.markdown("---")

# BOARD OPTIONS 
boards = [
    'CBSE', 'ICSE', 'CISCE', 'IB', 'NIOS',
    'Maharashtra State Board', 'Tamil Nadu State Board', 'Karnataka State Board',
    'Andhra Pradesh State Board', 'Telangana State Board', 'Kerala State Board',
    'West Bengal State Board', 'Gujarat State Board', 'Rajasthan State Board',
    'Madhya Pradesh State Board', 'Uttar Pradesh State Board', 'Bihar State Board',
    'Odisha State Board', 'Punjab State Board', 'Haryana State Board',
    'Jharkhand State Board', 'Chhattisgarh State Board', 'Assam State Board',
    'Jammu and Kashmir State Board', 'Himachal Pradesh State Board',
    'Uttarakhand State Board', 'Goa State Board', 'Tripura State Board',
    'Meghalaya State Board', 'Manipur State Board', 'Nagaland State Board',
    'Mizoram State Board', 'Arunachal Pradesh State Board', 'Sikkim State Board'
]

# USER INPUT FORM
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Your Details")

    board = st.selectbox("Board of Education", boards, index=0)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    percentage = st.number_input(
        "12th Grade Percentage (%)",
        min_value=35.0,
        max_value=100.0,
        value=75.0,
        step=0.1
    )

    math_score = st.number_input(
        "12th Math Score (out of 100)",
        min_value=35,
        max_value=100,
        value=80,
        step=1
    )

    st.markdown("---")

    if st.button("🔮 Predict My GMAT Score", type="primary", use_container_width=True):
        board_encoded = le_board.transform([board])[0]
        gender_encoded = le_gender.transform([gender])[0]

        features = np.array([[board_encoded, gender_encoded, percentage, math_score]])
        features_scaled = features.copy()
        features_scaled[:, 2:6] = scaler.transform(features[:, 2:6])  # scale percentage + scores

        gmat_pred = xgb_model.predict(features_scaled)[0]
        predicted_score = int(round(gmat_pred))
        predicted_score = max(200, min(800, predicted_score))  # GMAT score range

        st.session_state.prediction = {'score': predicted_score}

with col2:
    st.subheader("🎯 Prediction Results")

    if 'prediction' in st.session_state:
        pred = st.session_state.prediction

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
                    padding: 40px;
                    border-radius: 20px;
                    text-align: center;
                    color: white;
                    margin-bottom: 20px;'>
            <h1 style='font-size: 80px; margin: 0; font-weight: bold;'>{pred['score']}</h1>
            <h3 style='margin: 10px 0;'>Predicted GMAT Score</h3>
            <p style='margin: 0; opacity: 0.9;'>out of 800</p>
        </div>
        """, unsafe_allow_html=True)

        # GMAT Level
        if pred['score'] >= 700:
            level, color, desc = "Excellent", "#22c55e", "Top-tier GMAT performance"
        elif pred['score'] >= 650:
            level, color, desc = "Very Good", "#2563eb", "Competitive GMAT score"
        elif pred['score'] >= 600:
            level, color, desc = "Good", "#ca8a04", "Satisfactory for most programs"
        else:
            level, color, desc = "Needs Improvement", "#dc2626", "Consider improving quantitative/verbal skills"

        st.markdown(f"""
        <div style='background-color: {color}20;
                    border-left: 5px solid {color};
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;'>
            <h3 style='color: {color}; margin: 0;'>{level}</h3>
            <p style='color: {color}; margin: 5px 0 0 0; opacity: 0.8;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("🎯 **Predicted using XGBoost Model** — Accuracy depends on training data")

        score_min = max(200, pred['score'] - 20)
        score_max = min(800, pred['score'] + 20)
        st.success(f"📊 **Expected Score Range:** {score_min} - {score_max}")

        st.markdown("### 🎓 Typical GMAT Requirements")
        st.markdown("""
        - **Top Business Schools (Harvard, Stanford, Wharton):** 700–800  
        - **Good Schools:** 650–700  
        - **Most Programs:** 600–650  
        - **Minimum Requirement:** 500–600
        """)

    else:
        st.info("👈 Enter your details and click 'Predict' to see your estimated GMAT score!")

# FOOTER METRICS
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Model Accuracy", "71.0%", help="R² Score on test data")

with col_info2:
    st.metric("Training Data", "30,000", help="Student records used for training")

with col_info3:
    st.metric("Boards Supported", "34", help="Indian education boards")

st.markdown("---")
st.caption("Powered by Machine Learning • XGBoost Model")
