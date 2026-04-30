
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df['State_Name'] = df['State_Name'].str.lower().str.strip()
    df['Season'] = df['Season'].str.lower().str.strip()
    return df

df = load_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df[['level1','level2','level3','total','Yield']]
    y = df['Crop']

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=8,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(df)

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Smart Crop Recommendation System")
st.caption("Multi-cropping recommendation using ML")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("🌍 Select State", sorted(df['State_Name'].unique()))

with col2:
    season = st.selectbox("🌦️ Select Season", sorted(df['Season'].unique()))

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Get Crop Combination"):

    filtered = df[
        (df['State_Name'] == state) &
        (df['Season'] == season)
    ]

    if filtered.empty:
        st.warning("⚠️ No data available")
    else:
        avg_vals = filtered[['level1','level2','level3','total','Yield']].mean()

        input_data = [[
            avg_vals['level1'],
            avg_vals['level2'],
            avg_vals['level3'],
            avg_vals['total'],
            avg_vals['Yield']
        ]]

        probs = model.predict_proba(input_data)[0]
        top_idx = np.argsort(probs)[-3:][::-1]

        crops = model.classes_[top_idx]

        # -----------------------------
        # COMBINATION OUTPUT
        # -----------------------------
        st.success("🌱 Recommended Crop Combination")

        combo = " + ".join(crops)
        st.markdown(f"### 👉 {combo}")

        st.markdown("### 📊 Individual Confidence")
        for i, crop in enumerate(crops):
            st.write(f"👉 {crop} ({probs[top_idx[i]]:.2f})")

        st.info("💡 This combination is derived from top predicted crops suitable for multi-cropping based on historical data.")

# -----------------------------
# VISUALIZATION
# -----------------------------
st.markdown("## 📊 Crop Distribution Insights")

top_crops = df['Crop'].value_counts().head(10)

fig, ax = plt.subplots()
top_crops.plot(kind='bar', ax=ax)
ax.set_title("Top Crops")
st.pyplot(fig)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("ML-based Multi-Cropping Recommendation System")