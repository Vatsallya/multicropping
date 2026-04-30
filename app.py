import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Recommendation", layout="wide")

# -------------------------
# 🌿 CREATIVE BACKGROUND
# -------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-attachment: fixed;
}

/* light overlay for readability */
.main {
    background-color: rgba(255,255,255,0.88);
    padding: 20px;
    border-radius: 15px;
}

h1 {
    text-align: center;
    color: #1b5e20;
}

div.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

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
st.write(", ".join(sorted(filtered['Crop'].unique())))

# -------------------------
# YIELD (REAL)
# -------------------------
crop_yield = (
    filtered.groupby('Crop')['Yield']
    .mean()
    .sort_values(ascending=False)
)

# -------------------------
# GRAPH (ALL CROPS)
# -------------------------
st.subheader("📊 Crop Yield Comparison")

fig, ax = plt.subplots(figsize=(14, 6))

bars = ax.bar(crop_yield.index, crop_yield.values, color="#66bb6a")

# AXIS LABELS (YOUR REQUIREMENT)
ax.set_xlabel("Crop")
ax.set_ylabel("Average Yield (kg/ha)")

plt.xticks(rotation=90)

st.pyplot(fig)

# -------------------------
# BEST COMBINATION
# -------------------------
if st.button("Recommend Best Combination"):

    if len(crop_yield) < 2:
        st.error("Not enough crops")
    else:
        top1 = crop_yield.index[0]
        top2 = crop_yield.index[1]

        st.markdown(f"""
        <div style='padding:20px; border-radius:10px; background-color:#e6f4ea; text-align:center'>
            <h2>🌾 {top1} + {top2}</h2>
        </div>
        """, unsafe_allow_html=True)
