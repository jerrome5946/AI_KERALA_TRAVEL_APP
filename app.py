import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="Kerala AI Guide", page_icon="🌴", layout="wide")

# Mathematical Haversine function to calculate distances in kilometers
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Load core files
@st.cache_resource
def load_assets():
    model = joblib.load('models/kerala_model.joblib')
    encoder = joblib.load('models/vibe_encoder.joblib')
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    return model, encoder, config

try:
    model, encoder, config = load_assets()
    # FIXED: Cleaned up the path to point properly to your data folder
    df_places = pd.read_csv('data/data/kerala_spots.csv') 

    st.title("🌴 God's Own Country AI Planner & Live Guide")
    st.write("An upgraded system parsing contextual ML predictions and real-time proximity mappings.")
    st.divider()

    # Create Columns for different features
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Predictive Recommendation")
        budget = st.slider("💰 Your Budget (INR)", config['min_budget'], config['max_budget'], config['min_budget'], step=500)
        days = st.slider("📅 Trip Duration (Days)", 1, config['max_days'], 2)
        selected_vibe = st.selectbox("🎭 Select Preferred Vibe", config['vibes'])

        # FIXED: Integrated the weather feature smoothly inside this button block
        if st.button("🔮 Find My Destination", type="primary", use_container_width=True):
            encoded_vibe = encoder.transform([selected_vibe])[0]
            input_data = np.array([[budget, days, encoded_vibe]])
            prediction = model.predict(input_data)[0]
            st.success(f"### 📍 Recommended Spot: **{prediction}**")
            
            # LIVE WEATHER INTEGRATION
            weather_url = f"https://wttr.in/{prediction}?format=3"
            try:
                with st.spinner("Checking current climate conditions..."):
                    weather_response = requests.get(weather_url, timeout=5)
                    
                if weather_response.status_code == 200:
                    st.info(f"☀️ **Live Weather Feed:** {weather_response.text.strip()}")
                else:
                    st.caption("⚠️ Weather service temporarily busy.")
            except Exception:
                st.caption("🌐 Unable to load weather data (Check your internet connectivity).")

    with col2:
        st.header("📍 Nearest Destination Radar")
        st.write("Click below to fetch your current latitude and longitude values:")
        
        # User live GPS tracker button
        location = streamlit_geolocation()
        
        if location and location.get('latitude') is not None:
            user_lat = location['latitude']
            user_lon = location['longitude']
            
            st.info(f"**Current Coordinates Found:** Lat {user_lat:.4f}, Lon {user_lon:.4f}")
            
            # Distance computation loop
            df_places['Distance_KM'] = df_places.apply(
                lambda row: calculate_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
            )
            
            # Sort destinations by shortest distance
            df_sorted = df_places.sort_values(by='Distance_KM').head(3)
            
            st.subheader("🏁 Top 3 Nearest Spots From You:")
            for index, row in df_sorted.iterrows():
                st.write(f"👉 **{row['destination']}** ({row['vibe']}) — approx **{row['Distance_KM']:.1f} KM** away.")
            
            # Plot spots on a built-in interactive visual map layout
            st.write("### 🗺️ Proximity Map")
            map_data = df_sorted[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            st.map(map_data)
        else:
            st.warning("Awaiting location access tracker check. Please click the button module above.")

except Exception as e:
    st.error(f"Execution Error: {e}. Please make sure you have modified your datasets and re-run train.py.")