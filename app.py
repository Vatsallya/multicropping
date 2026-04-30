import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")

# -------------------------
# HEADER
# -------------------------
st.markdown("""
    <h1 style='text-align: center;'>🌾 Smart Crop Recommendation System</h1>
""", unsafe_allow_html=True)

st.markdown("---")

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

    df['State_Name'] = df['State_Name'].astype(str).str.lower().str.strip()
    df['Season'] = df['Season'].astype(str).str.lower().str.strip()
    df['Crop'] = df['Crop'].astype(str).str.strip()

    return df

df = load_data()

if df is None:
    st.error("Dataset missing required columns")
    st.stop()

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("📍 Select Inputs")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", sorted(df['State_Name'].unique()))

with col2:
    season = st.selectbox("Season", sorted(df['Season'].unique()))

st.markdown("---")

# -------------------------
# FILTER
# -------------------------
filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

if filtered.empty:
    st.warning("No data available for this selection")
    st.stop()

# -------------------------
# AVAILABLE CROPS
# -------------------------
st.subheader("🌱 Available Crops")

crop_list = sorted(filtered['Crop'].unique())

st.markdown(f"""
<div style='padding:10px; border-radius:10px; background-color:#f0f2f6'>
{", ".join(crop_list)}
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
st.subheader("📊 Crop Yield Analysis")

fig, ax = plt.subplots(figsize=(14, 5))

bars = ax.bar(crop_yield.index, crop_yield.values)

ax.set_xlabel("Crops")
ax.set_ylabel("Average Yield (kg/ha)")

plt.xticks(rotation=90)

# Highlight top 2
for i in range(min(2, len(bars))):
    bars[i].set_color('green')

st.pyplot(fig)

st.markdown("---")

# -------------------------
# BEST COMBINATION
# -------------------------
st.subheader("🏆 Best Crop Combination")

if st.button("Generate Recommendation"):

    if len(crop_yield) < 2:
        st.error("Not enough crops to form a combination")
    else:
        top1 = crop_yield.index[0]
        top2 = crop_yield.index[1]

        st.markdown(f"""
        <div style='padding:20px; border-radius:12px; background-color:#e6f4ea; text-align:center'>
            <h2>👉 {top1} + {top2}</h2>
        </div>
        """, unsafe_allow_html=True)
