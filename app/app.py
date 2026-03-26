from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import build_feature_frame, prepare_dataframe, preprocess
from src.recommender import recommend_games

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="iGaming Recommender",
    page_icon="🎰",
    layout="wide"
)
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
st.info("💡 Tip: Try high RTP + medium volatility for balanced recommendations.")
# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed.csv")
    return df

df = load_data()
df, feature_matrix, scaler, mappings = preprocess(df, return_mappings=True)

# -------------------------
# TITLE
# -------------------------
st.title("🎰 Smart iGaming Recommender System")
st.markdown("Discover casino games tailored to your risk and reward preferences.")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ User Preferences")

rtp = st.sidebar.slider("RTP (%)", 85.0, 99.5, 96.0)

volatility = st.sidebar.selectbox(
    "Volatility",
    list(mappings["volatility"].keys())
)

max_multiplier = st.sidebar.slider(
    "Max Multiplier",
    50, 5000, 500
)

min_bet = st.sidebar.slider(
    "Minimum Bet",
    0.1, 100.0, 1.0
)

bonus_buy = st.sidebar.selectbox("Bonus Buy Feature", [0, 1])
free_spins = st.sidebar.selectbox("Free Spins", [0, 1])

game_type = st.sidebar.selectbox(
    "Game Type",
    list(mappings["game_type"].keys())
)

user_preferences = pd.DataFrame(
    [
        {
            "rtp": rtp,
            "volatility": volatility,
            "max_multiplier": max_multiplier,
            "min_bet": min_bet,
            "bonus_buy_available": bonus_buy,
            "free_spins_feature": free_spins,
            "game_type": game_type,
        }
    ]
)

user_features = build_feature_frame(
    prepare_dataframe(user_preferences), mappings
)
user_scaled = scaler.transform(user_features)[0]

# -------------------------
# MAIN LAYOUT
# -------------------------
col1, col2 = st.columns([2, 1])

# -------------------------
# RECOMMENDATION
# -------------------------
with col1:
    if st.button("🎯 Recommend Games"):
        results = recommend_games(user_scaled, df, feature_matrix)

        st.subheader("Top Recommended Games")
        st.dataframe(results, use_container_width=True)

# -------------------------
# INSIGHTS PANEL
# -------------------------
with col2:
    st.subheader("📊 Dataset Insights")

    fig = px.scatter(
        df,
        x="rtp",
        y="max_multiplier",
        color="cluster",
        title="Game Clusters"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# EXTRA INSIGHTS
# -------------------------
st.subheader("📈 Feature Insights")

col3, col4 = st.columns(2)

with col3:
    fig2 = px.histogram(df, x="rtp", title="RTP Distribution")
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    fig3 = px.histogram(df, x="volatility", title="Volatility Distribution")
    st.plotly_chart(fig3, use_container_width=True)
