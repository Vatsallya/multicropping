import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")

# -------------------------
# 🌾 CUSTOM THEME (AGRI LOOK)
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #e8f5e9, #f1f8e9);
}

h1 {
    color: #2e7d32;
    text-align: center;
}

h2, h3 {
    color: #388e3c;
}

div.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #1b5e20;
}

.css-1d391kg {
    background-color: #f1f8e9;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("🌾 Smart Crop Recommendation System")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df.columns = df.columns.str.strip()

    if 'State' in df.columns:
        df.rename(columns={'State': 'State_Name'}, inplace=True)

    if 'Yield' not in df.columns:
        if 'Production' in df.columns and 'Area' in df.columns:
            df['Yield'] = df['Production'] / df['Area']
        else:
            return None

    df['State_Name'] = df['State_Name'].str.lower().str.strip()
    df['Season'] = df['Season'].str.lower().str.strip()
    df['Crop'] = df['Crop'].str.strip()

    return df

df = load_data()

if df is None:
    st.error("Dataset issue")
    st.stop()

# -------------------------
# INPUTS
# -------------------------
st.subheader("📍 Select Location & Season")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", sorted(df['State_Name'].unique()))

with col2:
    season = st.selectbox("Season", sorted(df['Season'].unique()))

# -------------------------
# FILTER
# -------------------------
filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

if filtered.empty:
    st.warning("No data available")
    st.stop()

# -------------------------
# AVAILABLE CROPS
# -------------------------
st.subheader("🌱 Available Crops")

st.markdown(f"""
<div style='padding:12px; background:#dcedc8; border-radius:10px'>
{", ".join(sorted(filtered['Crop'].unique()))}
</div>
""", unsafe_allow_html=True)

# -------------------------
# YIELD CALCULATION
# -------------------------
crop_yield = (
    filtered.groupby('Crop')['Yield']
    .mean()
    .sort_values(ascending=False)
)

# -------------------------
# GRAPH
# -------------------------
st.subheader("📊 Crop Yield Comparison")

fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(crop_yield.index, crop_yield.values, color="#66bb6a")

ax.set_xlabel("Crop")
ax.set_ylabel("Average Yield (kg/ha)")

plt.xticks(rotation=90)

st.pyplot(fig)

# -------------------------
# BEST COMBINATION
# -------------------------
st.subheader("🏆 Best Crop Combination")

if st.button("Recommend Best Combination"):

    if len(crop_yield) < 2:
        st.error("Not enough crops")
    else:
        top1 = crop_yield.index[0]
        top2 = crop_yield.index[1]

        st.markdown(f"""
        <div style='padding:20px; border-radius:12px; background-color:#c8e6c9; text-align:center'>
            <h2>🌾 {top1} + {top2}</h2>
        </div>
        """, unsafe_allow_html=True)
