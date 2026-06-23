import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Kerala AI Guide", page_icon="🌴", layout="centered")

# Load our offline trained models and configuration mappings
@st.cache_resource
def load_assets():
    model = joblib.load('models/kerala_model.joblib')
    encoder = joblib.load('models/vibe_encoder.joblib')
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    return model, encoder, config

try:
    model, encoder, config = load_assets()
    
    st.title("🌴 God's Own Country AI Planner")
    st.write("An offline, custom-trained Machine Learning model to match your perfect Kerala gateway.")
    st.divider()

    # User Input fields populated directly from your CSV bounds
    budget = st.slider("💰 Your Budget (INR)", config['min_budget'], config['max_budget'], config['min_budget'], step=500)
    days = st.slider("📅 Trip Duration (Days)", 1, config['max_days'], 2)
    selected_vibe = st.selectbox("🎭 What kind of vibe are you looking for?", config['vibes'])

    st.divider()

    if st.button("🔮 Find My Destination", type="primary", use_container_width=True):
        # 1. Transform the input vibe using our saved encoder
        encoded_vibe = encoder.transform([selected_vibe])[0]
        
        # 2. Structure the input array exactly like the training setup
        input_data = np.array([[budget, days, encoded_vibe]])
        
        # 3. Predict the best destination match
        prediction = model.predict(input_data)[0]
        
        st.success(f"### 📍 Recommended Spot: **{prediction}**")
        st.info(f"Pack your bags! Based on your budget of ₹{budget} and a preference for {selected_vibe} environments, {prediction} is your perfect match.")

except Exception as e:
    st.error("Assets not found. Please make sure you have created 'data/kerala_spots.csv' and executed 'train.py' first!")