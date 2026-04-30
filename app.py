import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# LOAD DATA
# -----------------------------
try:
    df_final = pd.read_csv("final_dataset.csv")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# -----------------------------
# CLEAN DATA
# -----------------------------
df_final['State_Name'] = df_final['State_Name'].str.lower().str.strip()
df_final['Season'] = df_final['Season'].str.lower().str.strip()

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

model = train_model(df_final)

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Crop Recommendation", layout="wide")
st.title("🌾 Smart Crop Recommendation System")

state = st.selectbox("Select State", sorted(df_final['State_Name'].unique()))
season = st.selectbox("Select Season", sorted(df_final['Season'].unique()))

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Crops"):
    try:
        filtered = df_final[
            (df_final['State_Name'] == state) &
            (df_final['Season'] == season)
        ]

        if filtered.empty:
            st.warning("⚠️ No data available for selected inputs")
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

            st.success("🌱 Recommended Crops:")
            for crop in crops:
                st.write("👉", crop)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")