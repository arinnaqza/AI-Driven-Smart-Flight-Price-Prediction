# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Flight Price Predictor (KRW)", layout="centered")

# --- CONFIG ---
MODEL_PATH = "xgboost_tuned_champion_model_krw.joblib"

# FEATURE COLUMNS - exact order from your X_train_krw.csv
FEATURE_COLUMNS = [
    'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
    'departure_time_Afternoon', 'departure_time_Early_Morning', 'departure_time_Evening', 'departure_time_Late_Night',
    'departure_time_Morning', 'departure_time_Night',
    'arrival_time_Afternoon', 'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night',
    'arrival_time_Morning', 'arrival_time_Night',
    'Route_Bangalore to Chennai', 'Route_Bangalore to Delhi', 'Route_Bangalore to Hyderabad', 'Route_Bangalore to Kolkata',
    'Route_Bangalore to Mumbai', 'Route_Chennai to Bangalore', 'Route_Chennai to Delhi', 'Route_Chennai to Hyderabad',
    'Route_Chennai to Kolkata', 'Route_Chennai to Mumbai', 'Route_Delhi to Bangalore', 'Route_Delhi to Chennai',
    'Route_Delhi to Hyderabad', 'Route_Delhi to Kolkata', 'Route_Delhi to Mumbai', 'Route_Hyderabad to Bangalore',
    'Route_Hyderabad to Chennai', 'Route_Hyderabad to Delhi', 'Route_Hyderabad to Kolkata', 'Route_Hyderabad to Mumbai',
    'Route_Kolkata to Bangalore', 'Route_Kolkata to Chennai', 'Route_Kolkata to Delhi', 'Route_Kolkata to Hyderabad',
    'Route_Kolkata to Mumbai', 'Route_Mumbai to Bangalore', 'Route_Mumbai to Chennai', 'Route_Mumbai to Delhi',
    'Route_Mumbai to Hyderabad', 'Route_Mumbai to Kolkata',
    'source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad',
    'source_city_Kolkata', 'source_city_Mumbai',
    'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad',
    'destination_city_Kolkata', 'destination_city_Mumbai',
    'class_Business', 'class_Economy',
    'Total_Stops', 'Duration_Minutes', 'Days_Left', 'Class_Encoded'
]

# Allowed categorical tokens inferred from feature names
AIRLINE_OPTIONS = {
    "AirAsia": "AirAsia",
    "Air India": "Air_India",
    "GO FIRST": "GO_FIRST",
    "Indigo": "Indigo",
    "SpiceJet": "SpiceJet",
    "Vistara": "Vistara"
}

CITY_OPTIONS = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]

DEPARTURE_TIME_OPTIONS = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
ARRIVAL_TIME_OPTIONS = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]

CLASS_OPTIONS = {"Economy": ("class_Economy", 0), "Business": ("class_Business", 1)}

STOP_MAPPING = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4}

# --- Helpers ---

def zeroed_feature_dict():
    """Return dict with all FEATURE_COLUMNS set to zero."""
    return {c: 0 for c in FEATURE_COLUMNS}

def set_onehot(prefix, token, features):
    """Set the corresponding one-hot feature for prefix_token if it exists in FEATURE_COLUMNS."""
    col_name = f"{prefix}_{token}"
    if col_name in features:
        features[col_name] = 1

def set_route_feature(src, dst, features):
    route_col = f"Route_{src} to {dst}"
    if route_col in features:
        features[route_col] = 1

def set_source_dest(src, dst, features):
    src_col = f"source_city_{src}"
    dst_col = f"destination_city_{dst}"
    if src_col in features:
        features[src_col] = 1
    if dst_col in features:
        features[dst_col] = 1

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'. Please run your training script to produce it.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- UI ---
st.title("✈️ Flight Price Predictor (KRW)")
st.markdown("---")
st.subheader("Simulate a Flight Booking")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        airline_readable = st.selectbox("Airline", list(AIRLINE_OPTIONS.keys()), index=list(AIRLINE_OPTIONS.keys()).index("Vistara"))
        source_city = st.selectbox("Departure City", CITY_OPTIONS, index=CITY_OPTIONS.index("Delhi"))
        destination_city = st.selectbox("Arrival City", CITY_OPTIONS, index=CITY_OPTIONS.index("Mumbai"))
        departure_time = st.selectbox("Departure Time Segment", DEPARTURE_TIME_OPTIONS, index=DEPARTURE_TIME_OPTIONS.index("Morning"))
        arrival_time = st.selectbox("Arrival Time Segment", ARRIVAL_TIME_OPTIONS, index=ARRIVAL_TIME_OPTIONS.index("Evening"))

    with col2:
        travel_class = st.selectbox("Travel Class", list(CLASS_OPTIONS.keys()), index=list(CLASS_OPTIONS.keys()).index("Economy"))
        days_left = st.slider("Days Left to Departure", min_value=1, max_value=180, value=45)
        total_stops_raw = st.selectbox("Total Stops", ['zero', 'one', 'two', 'three', 'four'], index=1)
        duration_minutes = st.number_input("Flight Duration (Minutes)", min_value=60, max_value=2000, value=300)

    submitted = st.form_submit_button("Predict Flight Price")

if submitted:
    # Build feature dict
    features = zeroed_feature_dict()

    # Airline
    airline_token = AIRLINE_OPTIONS[airline_readable]
    set_onehot("airline", airline_token, features)

    # Departure time and arrival time - convert names to match training tokens
    set_onehot("departure_time", departure_time, features)
    set_onehot("arrival_time", arrival_time, features)

    # Route, source, destination
    set_route_feature(source_city, destination_city, features)
    set_source_dest(source_city, destination_city, features)

    # Class one-hot and encoded
    class_col_name, class_encoded = CLASS_OPTIONS[travel_class]
    if class_col_name in features:
        features[class_col_name] = 1
    # Class_Encoded numeric
    if 'Class_Encoded' in features:
        features['Class_Encoded'] = class_encoded

    # Numeric features
    # Total_Stops
    total_stops_value = STOP_MAPPING.get(total_stops_raw, 0)
    features['Total_Stops'] = int(total_stops_value)

    # Duration and Days_Left
    features['Duration_Minutes'] = float(duration_minutes)
    features['Days_Left'] = int(days_left)

    # Final dataframe in exact column order
    input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Optional debug (uncomment to inspect)
    # st.write("Input dataframe (model features):")
    # st.dataframe(input_df)

    # Predict
    try:
        pred = model.predict(input_df.values)
        predicted_price = float(pred[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        mae = 24154.96  # if you want to show MAE context
        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric(label="Predicted Ticket Price", value=f"{predicted_price:,.0f} KRW", delta=f"MAE ±{mae:,.0f} KRW")

        # Recommendation text
        if days_left <= 7:
            st.warning("Booking within 7 days is high risk. Prices spike for last-minute tickets.")
        elif 30 <= days_left <= 60:
            st.success("You're in the optimal 30-60 day window. Good time to book.")
        elif 7 < days_left < 30:
            st.info("Slightly outside the prime window. Prices may still fluctuate.")
        else:
            st.info("Booking very early. Prices often drop closer to the 30-60 day window.")