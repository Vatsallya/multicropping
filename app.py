import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Recommendation", layout="wide")

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
# YIELD → SCORE (0–100)
# -------------------------
crop_yield = filtered.groupby('Crop')['Yield'].mean()

max_yield = crop_yield.max()

crop_score = (crop_yield / max_yield) * 100
crop_score = crop_score.sort_values(ascending=False)

# -------------------------
# TOP 10 GRAPH (CLEAN)
# -------------------------
top10 = crop_score.head(10)

st.subheader("📊 Top Crops Ranking (0–100 Score)")

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(top10.index, top10.values)

# Highlight best
bars[0].set_color('green')
if len(bars) > 1:
    bars[1].set_color('orange')

ax.set_ylabel("Score (0–100)")
plt.xticks(rotation=45)

st.pyplot(fig)

# -------------------------
# BEST COMBINATION
# -------------------------
if st.button("Recommend Best Combination"):

    if len(crop_score) < 2:
        st.error("Not enough crops")
    else:
        top1 = crop_score.index[0]
        top2 = crop_score.index[1]

        st.markdown(f"""
        <div style='padding:20px; border-radius:10px; background-color:#e6f4ea; text-align:center'>
            <h2>👉 {top1} + {top2}</h2>
        </div>
        """, unsafe_allow_html=True)
