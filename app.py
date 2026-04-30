import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("final_dataset.csv")

# Clean
df['State_Name'] = df['State_Name'].str.lower().str.strip()
df['Season'] = df['Season'].str.lower().str.strip()
df['Crop'] = df['Crop'].str.strip()

# -----------------------------
# ENCODING
# -----------------------------
le_state = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df['State_enc'] = le_state.fit_transform(df['State_Name'])
df['Season_enc'] = le_season.fit_transform(df['Season'])
df['Crop_enc'] = le_crop.fit_transform(df['Crop'])

# Features
X = df[['State_enc','Season_enc','level1','level2','level3','total','Yield']]
y = df['Crop_enc']

# -----------------------------
# TRAIN MODEL (LIGHTWEIGHT)
# -----------------------------
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X, y)

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Smart Crop Recommendation System")

state = st.selectbox("Select State", sorted(df['State_Name'].unique()))
season = st.selectbox("Select Season", sorted(df['Season'].unique()))

# -----------------------------
# FILTER DATA
# -----------------------------
filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

st.subheader("🌱 Crops available in selected conditions")

if filtered.empty:
    st.warning("No crops found")
else:
    st.write(", ".join(sorted(filtered['Crop'].unique())))

# -----------------------------
# PREDICTION
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

        # 🔥 restrict to only available crops
        available = filtered['Crop'].unique()
        available_idx = le_crop.transform(available)

        probs_filtered = [(i, probs[i]) for i in available_idx]
        probs_filtered = sorted(probs_filtered, key=lambda x: x[1], reverse=True)

        top2 = [le_crop.inverse_transform([i])[0] for i, _ in probs_filtered[:2]]

        st.success("🌾 Best Crop Combination")
        st.write(f"👉 {top2[0]} + {top2[1]}")