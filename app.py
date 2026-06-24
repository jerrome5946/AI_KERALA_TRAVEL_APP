import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from streamlit_geolocation import streamlit_geolocation

# 1. Clean, Minimal Page Config Configuration
st.set_page_config(
    page_title="Kerala Travel Planner", 
    page_icon="🌴", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Injecting Minimalist CSS Variables (No borders, subtle backgrounds)
# 2. Injecting Editorial Luxe Minimalist CSS
st.markdown("""
<style>
    /* Clean main page spacing */
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    
    /* Elegant luxury editorial typography */
    h1, h2, h3, h4 { 
        font-weight: 300 !important; 
        color: #2C2C2C !important; 
        letter-spacing: -0.02rem;
    }
    
    /* Subtle, premium, muted headers */
    h1 { font-size: 2.5rem !important; margin-bottom: 0.5rem !important; }
    
    /* Soft ivory panel cards with elegant thin borders instead of dark blocks */
    div[data-testid="stColumn"] {
        background-color: #EAE8E1;
        padding: 2.2rem;
        border-radius: 8px;
        border: 1px solid #DFDBD0;
    }
    
    /* Clean custom styling for native map frames */
    .stMap { border-radius: 6px; overflow: hidden; border: 1px solid #DFDBD0; }
</style>
""", unsafe_allow_html=True)

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

@st.cache_resource
def load_assets():
    model = joblib.load('models/kerala_model.joblib')
    encoder = joblib.load('models/vibe_encoder.joblib')
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    return model, encoder, config

try:
    model, encoder, config = load_assets()
    df_places = pd.read_csv('data/data/kerala_spots.csv') 

    # Clean App Title Bar
    st.title("Kerala Travel Planner 🌴")
    st.caption("Contextual ML destination tracking paired with geospatial location routing.")
    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Predictive Analytics")
        budget = st.slider("Budget Threshold (INR)", config['min_budget'], config['max_budget'], config['min_budget'], step=500)
        days = st.slider("Trip Scope (Days)", 1, config['max_days'], 2)
        selected_vibe = st.selectbox("Preferred Environmental Vibe", config['vibes'])

        st.write("") # Pure spacer layout padding
        
        if st.button("Generate Recommendation", type="primary", use_container_width=True):
            encoded_vibe = encoder.transform([selected_vibe])[0]
            input_data = np.array([[budget, days, encoded_vibe]])
            prediction = model.predict(input_data)[0]
            
            st.markdown(f"#### Target Match: **{prediction}**")
            
            # Weather fetch configuration block
            weather_url = f"https://wttr.in/{prediction}?format=3"
            try:
                weather_response = requests.get(weather_url, timeout=4)
                if weather_response.status_code == 200:
                    st.text(f"Climate Feed — {weather_response.text.strip()}")
            except Exception:
                pass
                
            predicted_row = df_places[df_places['destination'] == prediction]
            if not predicted_row.empty:
                p_lat = float(predicted_row['latitude'].values[0])
                p_lon = float(predicted_row['longitude'].values[0])
                
                if 'user_latitude' in st.session_state and st.session_state['user_latitude'] is not None:
                    u_lat = st.session_state['user_latitude']
                    u_lon = st.session_state['user_longitude']
                    pred_maps_url = f"https://www.google.com/maps/dir/?api=1&origin={u_lat},{u_lon}&destination={p_lat},{p_lon}&travelmode=driving"
                    st.link_button(f"Launch Navigation Map", pred_maps_url, use_container_width=True)

                pred_map_data = pd.DataFrame({'lat': [p_lat], 'lon': [p_lon]})
                st.map(pred_map_data, zoom=10)

    with col2:
        st.subheader("Proximity Radar")
        st.caption("Allow access to compute geographic distance thresholds natively via browser.")
        
        location = streamlit_geolocation()
        
        if location and location.get('latitude') is not None:
            user_lat = location['latitude']
            user_lon = location['longitude']
            
            st.session_state['user_latitude'] = user_lat
            st.session_state['user_longitude'] = user_lon
            
            df_places['Distance_KM'] = df_places.apply(
                lambda row: calculate_distance(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
            )
            df_sorted = df_places.sort_values(by='Distance_KM').head(3)
            
            st.write("") 
            st.markdown("##### Proximal Destinations")
            
            for index, row in df_sorted.iterrows():
                # Streamlined row element layout mapping
                col_info, col_btn = st.columns([2, 1])
                with col_info:
                    st.markdown(f"**{row['destination']}** \n*{row['vibe']} • {row['Distance_KM']:.1f} KM away*")
                with col_btn:
                    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={row['latitude']},{row['longitude']}&travelmode=driving"
                    st.link_button("Route Link", maps_url, use_container_width=True)
                st.divider()
            
            map_data = df_sorted[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            st.map(map_data)
        else:
            st.caption("Awaiting verification handshake query...")

except Exception as e:
    st.error(f"Execution Error: {e}")