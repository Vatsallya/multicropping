import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")

st.title("🌾 Smart Crop Recommendation System")

# -------------------------
# LOAD DATA (FAST)
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df.columns = df.columns.str.strip()

    # Fix column names if needed
    if 'State' in df.columns:
        df.rename(columns={'State': 'State_Name'}, inplace=True)

    # Create Yield if not present
    if 'Yield' not in df.columns:
        if 'Production' in df.columns and 'Area' in df.columns:
            df['Yield'] = df['Production'] / df['Area']
        else:
            return None

    # Clean data
    df['State_Name'] = df['State_Name'].astype(str).str.lower().str.strip()
    df['Season'] = df['Season'].astype(str).str.lower().str.strip()
    df['Crop'] = df['Crop'].astype(str).str.strip()

    return df

df = load_data()

if df is None:
    st.error("Dataset missing required columns")
    st.stop()

# -------------------------
# INPUTS
# -------------------------
col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("Select State", sorted(df['State_Name'].unique()))

with col2:
    season = st.selectbox("Select Season", sorted(df['Season'].unique()))

# -------------------------
# FILTER DATA
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
st.write(", ".join(sorted(filtered['Crop'].unique())))

# -------------------------
# YIELD CALCULATION (kg/ha)
# -------------------------
crop_yield = (
    filtered.groupby('Crop')['Yield']
    .mean()
    .sort_values(ascending=False)
)

# -------------------------
# GRAPH
# -------------------------
st.subheader("📊")

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(crop_yield.index, crop_yield.values)

ax.set_xlabel("Crops")
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

        st.success(f" {top1} + {top2}")
