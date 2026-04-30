import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# SAFE LOADING
# -----------------------------
try:
    model = joblib.load("model.pkl")
    le_state = joblib.load("le_state.pkl")
    le_season = joblib.load("le_season.pkl")
    le_crop = joblib.load("le_crop.pkl")
    df_final = pd.read_csv("final_dataset.csv")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Smart Crop Recommendation System")

# Normalize dataset (IMPORTANT FIX)
df_final['State_Name'] = df_final['State_Name'].str.lower().str.strip()
df_final['Season'] = df_final['Season'].str.lower().str.strip()

# Dropdowns
state = st.selectbox("Select State", sorted(df_final['State_Name'].unique()))
season = st.selectbox("Select Season", sorted(df_final['Season'].unique()))

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Crops"):
    try:
        filtered = df_final[
            (df_final['State_Name'] == state.lower().strip()) &
            (df_final['Season'] == season.lower().strip())
        ]

        if filtered.empty:
            st.warning("⚠️ No data available for selected inputs")
        else:
            avg_vals = filtered[['level1','level2','level3','total','Yield']].mean()

            state_enc = le_state.transform([state])[0]
            season_enc = le_season.transform([season])[0]

            input_data = [[
                state_enc,
                avg_vals['level1'],
                avg_vals['level2'],
                avg_vals['level3'],
                avg_vals['total'],
                avg_vals['Yield'],
                season_enc
            ]]

            # ⚠️ Ensure correct order (based on your training)
            # If your original order was:
            # [State_enc, Season_enc, level1, level2, level3, total, Yield]
            # then use THIS instead:

            input_data = [[
                state_enc,
                season_enc,
                avg_vals['level1'],
                avg_vals['level2'],
                avg_vals['level3'],
                avg_vals['total'],
                avg_vals['Yield']
            ]]

            probs = model.predict_proba(input_data)[0]
            top_idx = np.argsort(probs)[-3:][::-1]

            crops = le_crop.inverse_transform(top_idx)

            st.success("🌱 Recommended Crops:")
            for crop in crops:
                st.write("👉", crop)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
