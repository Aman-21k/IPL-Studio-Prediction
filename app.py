import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="IPL Winner Predictor",
    page_icon="🏏",
    layout="centered"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
CURRENT_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Punjab Kings",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore"
}

VENUE_MAP = {
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association Stadium, Mohali",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Uppal",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens"
}

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("ipl_winner_model_2024.pkl")
    label_encoder = joblib.load("winner_label_encoder_2024.pkl")
    features = joblib.load("model_features.pkl")
    return model, label_encoder, features


# --------------------------------------------------
# LOAD UI DATA (2024 ONLY)
# --------------------------------------------------
@st.cache_data
def load_ui_data():
    df = pd.read_csv("IPL_2008_to_2024_over_wise.csv")

    # -----------------------------
    # FILTER 2024 MATCHES ONLY
    # -----------------------------
    if "Season" in df.columns:
        df = df[df["Season"] == 2024]
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].dt.year == 2024]
    else:
        st.error("Dataset must contain Season or Date column")
        st.stop()

    # -----------------------------
    # SPLIT TEAMS
    # -----------------------------
    df[["Team1", "Team2"]] = df["Teams"].str.split(" vs ", expand=True)

    # -----------------------------
    # NORMALIZE TEAMS
    # -----------------------------
    for col in ["Team1", "Team2"]:
        df[col] = df[col].replace(TEAM_MAP)

    # -----------------------------
    # NORMALIZE VENUES
    # -----------------------------
    df["Venue"] = df["Venue"].replace(VENUE_MAP)

    # -----------------------------
    # CURRENT TEAMS ONLY
    # -----------------------------
    teams = sorted(
        t for t in set(df["Team1"]).union(df["Team2"])
        if t in CURRENT_TEAMS
    )

    # -----------------------------
    # 2024 VENUES ONLY
    # -----------------------------
    venues = sorted(df["Venue"].dropna().unique())

    return teams, venues


# --------------------------------------------------
# LOAD EVERYTHING
# --------------------------------------------------
model, le_target, FEATURES = load_model()
teams, venues = load_ui_data()

# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("🏏 IPL Match Winner Predictor")
st.caption(
    "Prediction based on pre-match data only "
    "(Teams, Venue, Toss)."
)

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    team2 = st.selectbox("Team 2", teams)

venue = st.selectbox("Venue", venues)

col3, col4 = st.columns(2)

with col3:
    toss_winner = st.selectbox(
        "Toss Winner",
        [team1, team2]
    )

with col4:
    toss_decision = st.selectbox(
        "Toss Decision",
        ["bat", "field"]
    )

st.divider()

# --------------------------------------------------
# PREDICTION LOGIC
# --------------------------------------------------
if st.button("🔮 Predict Winner", use_container_width=True):

    if team1 == team2:
        st.error("❌ Team 1 and Team 2 must be different.")
    else:
        input_df = pd.DataFrame({
            "Team1": [team1],
            "Team2": [team2],
            "Venue": [venue],
            "Toss_Winner": [toss_winner],
            "Toss_Decision": [toss_decision]
        })

        input_df = input_df[FEATURES]

        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)

        winner = le_target.inverse_transform(pred)[0]
        confidence = float(np.max(prob) * 100)

        st.success(f"🏆 **Predicted Winner:** {winner}")
        st.metric("Winning Probability", f"{confidence:.2f} %")
        st.progress(min(int(confidence), 100))

        st.caption(
            "⚠️ T20 cricket has high randomness. "
            "Predictions are probabilistic."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Built by Aman using CatBoost & Streamlit | "
    "Trained on IPL data (2008–2025)"
)
