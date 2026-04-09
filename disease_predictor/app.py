import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
    }
    .prediction-box {
        background-color: #f0f7ff;
        border-left: 5px solid #1f77b4;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_path = 'models/disease_model.pkl'
        symptoms_path = 'models/symptom_names.pkl'
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()
        if not os.path.exists(symptoms_path):
            st.error(f"Symptoms file not found: {symptoms_path}")
            st.stop()
            
        model = joblib.load(model_path)
        symptoms = joblib.load(symptoms_path)
        return model, symptoms
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, symptoms = load_model()

st.markdown('<div class="main-header">🏥 Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown("AI-Powered Disease Detection Based on Symptoms", unsafe_allow_html=True)

st.warning("⚠️ DISCLAIMER: Educational tool only. Consult a healthcare professional.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select:", ["Predict", "About", "Symptoms"])

disease_info = {
    "COVID-19": {
        "description": "A viral respiratory infection caused by SARS-CoV-2.",
        "common": "Very common"
    },
    "Flu": {
        "description": "A seasonal influenza infection with fever, cough, and body aches.",
        "common": "Common"
    },
    "Malaria": {
        "description": "A mosquito-borne disease with fever and chills.",
        "common": "Common in tropical regions"
    },
    "Pneumonia": {
        "description": "Lung infection that causes cough, fever, and breathing trouble.",
        "common": "Common"
    },
    "Typhoid": {
        "description": "Bacterial infection causing high fever and abdominal pain.",
        "common": "Less common"
    },
    "Food Poisoning": {
        "description": "Illness from contaminated food causing nausea and diarrhea.",
        "common": "Common"
    },
    "Asthma": {
        "description": "Chronic lung condition that causes wheezing and breathing difficulty.",
        "common": "Very common"
    },
    "Allergies": {
        "description": "Immune response to allergens causing sneezing and itching.",
        "common": "Very common"
    }
}

available_info = {name: info for name, info in disease_info.items() if name in model.classes_}

st.sidebar.subheader("Disease Info")
selected_info = st.sidebar.selectbox(
    "Choose a disease for a short description:",
    ["Select one"] + list(available_info.keys())
)

if selected_info != "Select one":
    info = available_info[selected_info]
    st.sidebar.write(f"**Commonness:** {info['common']}")
    st.sidebar.write(info["description"])

if page == "Predict":
    st.header("Enter Your Symptoms")
    
    input_method = st.radio("Select symptoms by:", ["Checkbox", "Numbers"])
    
    selected_symptoms = []
    
    if input_method == "Checkbox":
        st.subheader("Select Your Symptoms:")
        cols = st.columns(3)
        
        for idx, symptom in enumerate(symptoms):
            col = cols[idx % 3]
            if col.checkbox(symptom, key=f"cb_{idx}"):
                selected_symptoms.append(symptom)
    
    else:
        st.subheader("Enter Symptom Numbers:")
        symptom_df = pd.DataFrame({
            "No.": range(1, len(symptoms) + 1),
            "Symptom": symptoms
        })
        st.dataframe(symptom_df, use_container_width=True)
        
        numbers_input = st.text_input("Numbers (comma-separated, e.g., 1,2,5):")
        
        if numbers_input:
            try:
                numbers = [int(x.strip()) - 1 for x in numbers_input.split(',')]
                selected_symptoms = [symptoms[i] for i in numbers if 0 <= i < len(symptoms)]
            except:
                st.error("Invalid input!")
    
    if selected_symptoms:
        st.success(f"✓ Selected {len(selected_symptoms)} symptom(s)")
    
    if st.button("🔍 Predict Disease"):
        if not selected_symptoms:
            st.error("Select at least one symptom!")
        else:
            feature_vector = np.array([1 if s in selected_symptoms else 0 for s in symptoms]).reshape(1, -1)
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            confidence = max(probabilities) * 100
            
            st.markdown(f'<div class="prediction-box"><h2>🎯 {prediction}</h2><h3>Confidence: {confidence:.1f}%</h3></div>', unsafe_allow_html=True)
            
            st.subheader("Disease Probabilities")
            disease_probs = pd.DataFrame({
                'Disease': model.classes_,
                'Probability %': probabilities * 100
            }).sort_values('Probability %', ascending=False)
            
            st.dataframe(disease_probs, use_container_width=True)
            st.bar_chart(disease_probs.set_index('Disease')['Probability %'].head(10))

elif page == "About":
    st.header("About This System")
    st.write(f"**Total Symptoms:** {len(symptoms)}")
    st.write(f"**Total Diseases:** {len(model.classes_)}")
    st.write("**Algorithm:** Random Forest Classifier")
    st.info("Educational tool - NOT for medical diagnosis")

else:
    st.header("Available Symptoms")
    symptoms_df = pd.DataFrame({
        'No.': range(1, len(symptoms) + 1),
        'Symptom': symptoms
    })
    st.dataframe(symptoms_df, use_container_width=True)