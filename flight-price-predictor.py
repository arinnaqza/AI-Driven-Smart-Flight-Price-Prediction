# ------------------------------------------------------------
# Streamlit App – Flight Price Predictor (KRW)
# Dark blue + white theme with Poppins
# ------------------------------------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# PAGE CONFIG + THEME
# ------------------------------------------------------------
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}
.stApp {
    background: radial-gradient(circle at 20% 20%, #0d1b34 0%, #081124 30%, #040a18 60%, #020611 100%);
    color: #f5f7fb;
}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
}
.result-card {
    background: linear-gradient(135deg, #10254d, #0c1b37);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
    text-align: center;
}
.price-text { font-size: 32px; font-weight: 700; color: #fefefe; }
.price-subtext { font-size: 13px; color: #d4dbeb; }
.info-box {
    margin-top: 12px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
}
.big-title { font-size: 32px; font-weight: 700; letter-spacing: 0.04em; }
.sub { font-size: 15px; color: #d4dbeb; }
.stButton>button {
    border-radius: 999px;
    padding: 0.7rem 1.4rem;
    border: none;
    font-weight: 600;
    color: #0b1930;
    background: linear-gradient(135deg, #f6f8ff, #dbe5ff);
    box-shadow: 0 10px 26px rgba(0,0,0,0.4);
}
.stButton>button:hover { filter: brightness(1.05); }
.stSelectbox>div>div, .stNumberInput input {
    background: rgba(255,255,255,0.06) !important;
    color: #f5f7fb !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
.stSlider [role="slider"] { background: #dbe5ff !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
        <div class="big-title">✈️ Flight Price Predictor</div>
        <div class="sub">Estimate fares in KRW with smart booking insights</div>
    </div>
    <div class="sub">XGBoost Model • KRW</div>
</div>
<hr style="border:0;border-top:1px solid rgba(255,255,255,0.15); margin-bottom: 12px;">
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# MODEL + CONSTANTS
# ------------------------------------------------------------
MODEL_FILE = "xgboost_tuned_champion_model_krw.joblib"
MODEL_MAE = 24154.96  # reference MAE (KRW)

FEATURE_COLUMNS = [
    "airline_AirAsia",
    "airline_Air_India",
    "airline_GO_FIRST",
    "airline_Indigo",
    "airline_SpiceJet",
    "airline_Vistara",
    "departure_time_Afternoon",
    "departure_time_Early_Morning",
    "departure_time_Evening",
    "departure_time_Late_Night",
    "departure_time_Morning",
    "departure_time_Night",
    "arrival_time_Afternoon",
    "arrival_time_Early_Morning",
    "arrival_time_Evening",
    "arrival_time_Late_Night",
    "arrival_time_Morning",
    "arrival_time_Night",
    "Route_Bangalore to Chennai",
    "Route_Bangalore to Delhi",
    "Route_Bangalore to Hyderabad",
    "Route_Bangalore to Kolkata",
    "Route_Bangalore to Mumbai",
    "Route_Chennai to Bangalore",
    "Route_Chennai to Delhi",
    "Route_Chennai to Hyderabad",
    "Route_Chennai to Kolkata",
    "Route_Chennai to Mumbai",
    "Route_Delhi to Bangalore",
    "Route_Delhi to Chennai",
    "Route_Delhi to Hyderabad",
    "Route_Delhi to Kolkata",
    "Route_Delhi to Mumbai",
    "Route_Hyderabad to Bangalore",
    "Route_Hyderabad to Chennai",
    "Route_Hyderabad to Delhi",
    "Route_Hyderabad to Kolkata",
    "Route_Hyderabad to Mumbai",
    "Route_Kolkata to Bangalore",
    "Route_Kolkata to Chennai",
    "Route_Kolkata to Delhi",
    "Route_Kolkata to Hyderabad",
    "Route_Kolkata to Mumbai",
    "Route_Mumbai to Bangalore",
    "Route_Mumbai to Chennai",
    "Route_Mumbai to Delhi",
    "Route_Mumbai to Hyderabad",
    "Route_Mumbai to Kolkata",
    "source_city_Bangalore",
    "source_city_Chennai",
    "source_city_Delhi",
    "source_city_Hyderabad",
    "source_city_Kolkata",
    "source_city_Mumbai",
    "destination_city_Bangalore",
    "destination_city_Chennai",
    "destination_city_Delhi",
    "destination_city_Hyderabad",
    "destination_city_Kolkata",
    "destination_city_Mumbai",
    "class_Business",
    "class_Economy",
    "Total_Stops",
    "Duration_Minutes",
    "Days_Left",
    "Class_Encoded",
]

NUMERIC_MEANS = {
    "Duration_Minutes": 12.077626670199082,
    "Days_Left": 26.05910404473153,
    "Total_Stops": 0.8744923605712752,
}
NUMERIC_STDS = {
    "Duration_Minutes": 7.150186744214485,
    "Days_Left": 13.552063172739185,
    "Total_Stops": 0.33129363391673256,
}

AIRLINE_TOKENS = {
    "AirAsia": "AirAsia",
    "Air India": "Air_India",
    "GO FIRST": "GO_FIRST",
    "Indigo": "Indigo",
    "SpiceJet": "SpiceJet",
    "Vistara": "Vistara",
}

AIRLINES = list(AIRLINE_TOKENS.keys())
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
TIME_SEGMENTS = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
CLASSES = ["Economy", "Business"]
STOP_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4}
CLASS_OPTIONS = {"Economy": ("class_Economy", 0), "Business": ("class_Business", 1)}
NO_BUSINESS_AIRLINES = {"AirAsia", "GO FIRST", "Indigo", "SpiceJet"}

# Session defaults for class/airline handling
if "travel_class_out" not in st.session_state:
    st.session_state.travel_class_out = "Economy"
if "travel_class_ret" not in st.session_state:
    st.session_state.travel_class_ret = "Economy"
if "prev_airline" not in st.session_state:
    st.session_state.prev_airline = None


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def zeroed_feature_dict():
    return {c: 0 for c in FEATURE_COLUMNS}


def set_onehot(prefix, token, features):
    col = f"{prefix}_{token}"
    if col in features:
        features[col] = 1


def set_route_feature(src, dst, features):
    col = f"Route_{src} to {dst}"
    if col in features:
        features[col] = 1


def set_source_dest(src, dst, features):
    if f"source_city_{src}" in features:
        features[f"source_city_{src}"] = 1
    if f"destination_city_{dst}" in features:
        features[f"destination_city_{dst}"] = 1


def standardize_numeric(v, key):
    mean = NUMERIC_MEANS[key]
    std = NUMERIC_STDS[key]
    return (v - mean) / std if std else v


def build_features(airline, source, dest, departure_time, arrival_time, travel_class, days_left, stops_key, duration_minutes):
    final = zeroed_feature_dict()
    set_onehot("airline", AIRLINE_TOKENS[airline], final)
    set_onehot("departure_time", departure_time, final)
    set_onehot("arrival_time", arrival_time, final)
    set_route_feature(source, dest, final)
    set_source_dest(source, dest, final)
    class_col, class_encoded = CLASS_OPTIONS[travel_class]
    final[class_col] = 1
    final["Class_Encoded"] = class_encoded
    final["Total_Stops"] = standardize_numeric(STOP_MAP[stops_key], "Total_Stops")
    final["Duration_Minutes"] = standardize_numeric(float(duration_minutes) / 60, "Duration_Minutes")
    final["Days_Left"] = standardize_numeric(float(days_left), "Days_Left")
    return pd.DataFrame([final], columns=FEATURE_COLUMNS)


def booking_tip(price, days_left):
    tip = ""
    if days_left <= 3:
        tip = "Very close to departure. Prices commonly spike."
    elif days_left <= 7:
        tip = "Short booking window; fare drops are unlikely."
    elif days_left <= 20:
        tip = "Fares fluctuate in this zone. Monitoring could help."
    elif days_left <= 40:
        tip = "Typical period for decent fare opportunities."
    elif days_left <= 70:
        tip = "Historically strong booking zone."
    else:
        tip = "Early booking; fare patterns vary by airline."

    if price > 400000:
        tip += " Estimate is on the higher side, flexible dates may help."
    elif price < 200000:
        tip += " Estimate is relatively low compared to similar routes."
    return tip


def time_suggestion(label, seg):
    cheap = {"Early_Morning", "Late_Night"}
    mid = {"Morning", "Afternoon"}
    pretty = seg.replace("_", " ")
    if seg in cheap:
        return f"{label}: {pretty} often falls in lower fare ranges."
    if seg in mid:
        return f"{label}: {pretty} is typically mid priced."
    return f"{label}: {pretty} often trends higher."


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
if not os.path.exists(MODEL_FILE):
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()

model = joblib.load(MODEL_FILE)

# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            airline = st.selectbox("Outbound Airline", AIRLINES)
            source = st.selectbox("Departure City", CITIES, index=CITIES.index("Delhi"))
            dest = st.selectbox("Arrival City", CITIES, index=CITIES.index("Mumbai"))
        with c2:
            departure_time = st.selectbox("Departure Time", TIME_SEGMENTS, index=TIME_SEGMENTS.index("Morning"))
            arrival_time = st.selectbox("Arrival Time", TIME_SEGMENTS, index=TIME_SEGMENTS.index("Evening"))

        c3, c4, c5 = st.columns(3)
        with c3:
            supports_business = airline not in NO_BUSINESS_AIRLINES
            class_options = CLASSES if supports_business else ["Economy"]
            # Reset class selection if airline changed or previous choice is invalid
            if st.session_state.prev_airline != airline or st.session_state.travel_class_out not in class_options:
                st.session_state.travel_class_out = class_options[0]
            travel_class = st.selectbox(
                "Travel Class",
                class_options,
                index=class_options.index(st.session_state.travel_class_out),
                key="travel_class_out",
            )
            st.session_state.prev_airline = airline
            notice = st.empty()
            if not supports_business:
                notice.caption("This airline is economy-only; business class is not available.")
        with c4:
            days_left = st.slider("Days Left", 1, 180, 45)
        with c5:
            stops = st.selectbox("Total Stops", list(STOP_MAP.keys()), index=1)

        duration = st.number_input("Flight Duration (minutes)", min_value=60, max_value=2000, value=300)

        st.markdown("---")
        round_trip = st.checkbox("Estimate round trip (two-way) with a return leg?")

        if round_trip:
            st.markdown("**Return Leg Inputs**")
            rc1, rc2 = st.columns(2)
            with rc1:
                ret_airline = st.selectbox("Return Airline", AIRLINES, index=AIRLINES.index(airline))
                ret_source = st.selectbox("Return Departure City", CITIES, index=CITIES.index(dest))
                ret_dest = st.selectbox("Return Arrival City", CITIES, index=CITIES.index(source))
            with rc2:
                ret_departure_time = st.selectbox("Return Departure Time", TIME_SEGMENTS, index=TIME_SEGMENTS.index("Afternoon"))
                ret_arrival_time = st.selectbox("Return Arrival Time", TIME_SEGMENTS, index=TIME_SEGMENTS.index("Evening"))

            rc3, rc4, rc5 = st.columns(3)
            with rc3:
                ret_days_left = st.slider("Days Left Until Return", 1, 365, 7)
            with rc4:
                ret_stops = st.selectbox("Return Total Stops", list(STOP_MAP.keys()), index=1)
            with rc5:
                ret_duration = st.number_input("Return Flight Duration (minutes)", min_value=60, max_value=2000, value=300, key="ret_duration")

            # Return travel class (can differ from outbound)
            supports_business_ret = ret_airline not in NO_BUSINESS_AIRLINES
            ret_class_options = CLASSES if supports_business_ret else ["Economy"]
            if st.session_state.prev_airline != ret_airline or st.session_state.travel_class_ret not in ret_class_options:
                st.session_state.travel_class_ret = ret_class_options[0]
            ret_travel_class = st.selectbox(
                "Return Travel Class",
                ret_class_options,
                index=ret_class_options.index(st.session_state.travel_class_ret),
                key="travel_class_ret",
            )
            if not supports_business_ret:
                st.caption("Return leg: this airline is economy-only; business class is not available.")

        submit = st.form_submit_button("Run Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if submit:
        invalid = False
        if source == dest:
            st.error("Departure and arrival cities must be different for the outbound leg.")
            invalid = True
        if round_trip:
            if ret_source == ret_dest:
                st.error("Departure and arrival cities must be different for the return leg.")
                invalid = True
        if invalid:
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        outbound_class = st.session_state.travel_class_out
        return_class = st.session_state.travel_class_ret if round_trip else outbound_class

        outbound_df = build_features(
            airline, source, dest, departure_time, arrival_time, outbound_class, days_left, stops, duration
        )
        outbound_price = float(model.predict(outbound_df.values)[0])

        ret_price = None
        if round_trip:
            ret_df = build_features(
                ret_airline, ret_source, ret_dest, ret_departure_time, ret_arrival_time, return_class, ret_days_left, ret_stops, ret_duration
            )
            ret_price = float(model.predict(ret_df.values)[0])
            total_price = outbound_price + ret_price
        else:
            total_price = outbound_price

        st.markdown(
            f"""
            <div class="result-card">
                <div class="price-text">{total_price:,.0f} KRW</div>
                <div class="price-subtext">{'Estimated round trip fare' if round_trip else 'Estimated one way fare'}</div>
                {"<div class='price-subtext'>Outbound ~ " + format(outbound_price, ',.0f') + " KRW" + (" | Return ~ " + format(ret_price, ',.0f') + " KRW" if ret_price is not None else "") + "</div>" if round_trip else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Model MAE ≈ {MODEL_MAE:,.0f} KRW")

        outbound_tip = booking_tip(outbound_price, days_left)
        return_tip = booking_tip(ret_price, ret_days_left) if ret_price is not None else None

        if round_trip and return_tip:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Best Booking Time (Outbound)</strong><br>
                    ▸ {outbound_tip}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Best Booking Time (Return)</strong><br>
                    ▸ {return_tip}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Best Booking Time</strong><br>
                    ▸ {outbound_tip}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Airline price comparison (percent cheaper) per leg
        def best_airline_for_leg(src, dst, dep_t, arr_t, cls, days, stops_key, dur):
            prices = []
            for cand in AIRLINES:
                if cls == "Business" and cand in NO_BUSINESS_AIRLINES:
                    continue
                cand_price = float(
                    model.predict(
                        build_features(
                            cand, src, dst, dep_t, arr_t, cls, days, stops_key, dur
                        ).values
                    )[0]
                )
                prices.append((cand, cand_price))
            if not prices:
                return None, None, None
            prices.sort(key=lambda x: x[1])
            best_airline, best_price = prices[0]
            second_price = prices[1][1] if len(prices) > 1 else None
            return best_airline, best_price, second_price

        best_out_airline, best_out_price, second_out_price = best_airline_for_leg(
            source, dest, departure_time, arrival_time, outbound_class, days_left, stops, duration
        )
        best_ret_airline = best_ret_price = second_ret_price = None
        if round_trip:
            best_ret_airline, best_ret_price, second_ret_price = best_airline_for_leg(
                ret_source, ret_dest, ret_departure_time, ret_arrival_time, return_class, ret_days_left, ret_stops, ret_duration
            )

        budget_airlines = {"AirAsia", "SpiceJet", "GO FIRST", "Indigo"}
        premium_airlines = {"Vistara", "Air India"}

        def airline_note(name):
            if name in budget_airlines:
                return f"{name} is budget-friendly. Staying with low-cost carriers like {', '.join(sorted(budget_airlines))} usually keeps fares lower."
            if name in premium_airlines:
                return f"{name} is a full-service carrier and often prices higher. For cheaper options, try budget airlines such as {', '.join(sorted(budget_airlines))}."
            return f"If price is the priority, low-cost carriers ({', '.join(sorted(budget_airlines))}) tend to be cheaper than premium airlines ({', '.join(sorted(premium_airlines))})."

        def percent_line(chosen_airline, chosen_price, best_airline, best_price, second_price, label):
            if best_airline is None or best_price is None:
                return ""
            if best_airline == chosen_airline:
                if second_price:
                    pct = (second_price - best_price) / second_price * 100 if second_price else 0
                    return f"{label}: {chosen_airline} is ~{pct:,.1f}% cheaper than the next-best option for this leg."
                return f"{label}: {chosen_airline} is the cheapest option for this leg."
            pct = (chosen_price - best_price) / best_price * 100 if best_price else 0
            return f"{label}: {chosen_airline} is ~{pct:,.1f}% higher than {best_airline} for this leg."

        outbound_percent = percent_line(airline, outbound_price, best_out_airline, best_out_price, second_out_price, "Outbound")
        return_percent = ""
        if round_trip:
            return_percent = percent_line(ret_airline, ret_price, best_ret_airline, best_ret_price, second_ret_price, "Return")

        if round_trip:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Best Airline Options (Outbound)</strong><br>
                    ▸ Outbound: {airline_note(airline)}<br>
                    {("▸ " + outbound_percent) if outbound_percent else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Best Airline Options (Return)</strong><br>
                    ▸ Return: {airline_note(ret_airline)}<br>
                    {("▸ " + return_percent) if return_percent else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Best Airline Options</strong><br>
                    ▸ Outbound: {airline_note(airline)}<br>
                    {("▸ " + outbound_percent) if outbound_percent else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if round_trip and ret_price is not None:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Perfect Flight Time (Outbound)</strong><br>
                    ▸ {time_suggestion("Departure", departure_time)}<br>
                    ▸ {time_suggestion("Arrival", arrival_time)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Perfect Flight Time (Return)</strong><br>
                    ▸ {time_suggestion("Departure", ret_departure_time)}<br>
                    ▸ {time_suggestion("Arrival", ret_arrival_time)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="info-box">
                    <strong>Your Perfect Flight Time</strong><br>
                    ▸ {time_suggestion("Departure", departure_time)}<br>
                    ▸ {time_suggestion("Arrival", arrival_time)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
