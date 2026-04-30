import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Smart Crop Recommendation", layout="centered")

# -----------------------------
# CACHE DATA + MODEL (SPEED)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df['State_Name'] = df['State_Name'].str.lower().str.strip()
    df['Season'] = df['Season'].str.lower().str.strip()
    df['Crop'] = df['Crop'].str.strip()
    return df

@st.cache_resource
def train_model(df):
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    df['State_enc'] = le_state.fit_transform(df['State_Name'])
    df['Season_enc'] = le_season.fit_transform(df['Season'])
    df['Crop_enc'] = le_crop.fit_transform(df['Crop'])

    X = df[['State_enc','Season_enc','level1','level2','level3','total','Yield']]
    y = df['Crop_enc']

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    return model, le_state, le_season, le_crop

df = load_data()
model, le_state, le_season, le_crop = train_model(df)

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Smart Crop Recommendation System")

state = st.selectbox("Select State", sorted(df['State_Name'].unique()))
season = st.selectbox("Select Season", sorted(df['Season'].unique()))

# -----------------------------
# FILTER
# -----------------------------
filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

st.subheader("🌱 Available Crops")
if filtered.empty:
    st.warning("No data available for this selection")
else:
    st.write(", ".join(sorted(filtered['Crop'].unique())))

# -----------------------------
# RECOMMENDATION
# -----------------------------
if st.button("Recommend Best Combination"):

    if filtered.empty:
        st.error("No data for prediction")
    else:
        avg_vals = filtered[['level1','level2','level3','total','Yield']].mean()

        input_data = [[
            le_state.transform([state])[0],
            le_season.transform([season])[0],
            avg_vals['level1'],
            avg_vals['level2'],
            avg_vals['level3'],
            avg_vals['total'],
            avg_vals['Yield']
        ]]

        probs = model.predict_proba(input_data)[0]

        # Restrict to valid crops only
        available = filtered['Crop'].unique()
        available_idx = le_crop.transform(available)

        crop_probs = [(le_crop.inverse_transform([i])[0], probs[i]) for i in available_idx]
        crop_probs = sorted(crop_probs, key=lambda x: x[1], reverse=True)

        # Top 2 crops
        crop1, p1 = crop_probs[0]
        crop2, p2 = crop_probs[1]

        # -----------------------------
        # CONFIDENCE CALCULATION
        # -----------------------------
        p1_pct = round(p1 * 100, 2)
        p2_pct = round(p2 * 100, 2)

        # Combined score (simple average)
        combined = round(((p1 + p2) / 2) * 100, 2)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.success("🌾 Best Crop Combination")

        st.markdown(f"### 👉 {crop1} + {crop2}")

        st.subheader("📊 Success Probability")

        st.write(f"🌱 **{crop1}** → {p1_pct}%")
        st.write(f"🌱 **{crop2}** → {p2_pct}%")
        st.write(f"🌾 **Combined Success** → {combined}%")

        # Visual bars (fast UI)
        st.progress(p1)
        st.progress(p2)