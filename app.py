import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crop Recommendation", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df.columns = df.columns.str.strip()

    # Fix column name if needed
    if 'State' in df.columns:
        df.rename(columns={'State': 'State_Name'}, inplace=True)

    # Create Yield if not present
    if 'Yield' not in df.columns:
        if 'Production' in df.columns and 'Area' in df.columns:
            df['Yield'] = df['Production'] / df['Area']
        else:
            st.error("Dataset must contain Yield or (Production & Area)")
            st.stop()

    # Clean text
    df['State_Name'] = df['State_Name'].astype(str).str.lower().str.strip()
    df['Season'] = df['Season'].astype(str).str.lower().str.strip()
    df['Crop'] = df['Crop'].astype(str).str.strip()

    return df

df = load_data()

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Smart Crop Recommendation System")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("Select State", sorted(df['State_Name'].unique()))

with col2:
    season = st.selectbox("Select Season", sorted(df['Season'].unique()))

# -----------------------------
# FILTER DATA
# -----------------------------
filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

st.subheader("🌱 Available Crops")

if filtered.empty:
    st.warning("No data available for this selection")
    st.stop()

st.write(", ".join(sorted(filtered['Crop'].unique())))

# -----------------------------
# YIELD CALCULATION
# -----------------------------
crop_yield = filtered.groupby('Crop')['Yield'].mean().sort_values(ascending=False)

# -----------------------------
# GRAPH (ALL CROPS)
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(crop_yield.index, crop_yield.values)
ax.set_xlabel("Crops")
ax.set_ylabel("Average Yield (kg/ha)")

plt.xticks(rotation=90)

st.pyplot(fig)

# -----------------------------
# RECOMMENDATION
# -----------------------------
if st.button("Recommend Best Combination"):

    if len(crop_yield) < 2:
        st.error("Not enough crops for recommendation")
    else:
        best_two = crop_yield.head(2).index.tolist()

        st.success(f"👉 {best_two[0]} + {best_two[1]}")
