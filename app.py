import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Crop Recommendation", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df['State_Name'] = df['State_Name'].str.lower().str.strip()
    df['Season'] = df['Season'].str.lower().str.strip()
    df['Crop'] = df['Crop'].str.strip()
    return df

df = load_data()

# -----------------------------
# TRAIN MODEL (NO PKL)
# -----------------------------
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

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_state, le_season, le_crop

model, le_state, le_season, le_crop = train_model(df)

# -----------------------------
# UI
# -----------------------------
st.markdown("<h1 style='text-align:center;'>Smart Crop Recommendation</h1><hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("Select State", sorted(df['State_Name'].unique()))

with col2:
    season = st.selectbox("Select Season", sorted(df['Season'].unique()))

filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

st.subheader("Available Crops")

if filtered.empty:
    st.warning("No data available")
else:
    st.write(", ".join(sorted(filtered['Crop'].unique())))

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Recommend Best Combination"):

    if filtered.empty:
        st.error("No data available")

    else:
        # Average features
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

        # -----------------------------
        # Yield (for stability)
        # -----------------------------
        crop_yield = filtered.groupby('Crop')['Yield'].mean()

        scores = []

        for crop in crop_yield.index:
            if crop not in le_crop.classes_:
                continue

            idx = le_crop.transform([crop])[0]
            model_prob = probs[idx]
            yield_score = crop_yield[crop] / crop_yield.max()

            final_score = 0.6 * model_prob + 0.4 * yield_score

            scores.append((crop, final_score))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # -----------------------------
        # GRAPH
        # -----------------------------
        plot_df = pd.DataFrame(scores[:15], columns=["Crop", "Score"])

        st.subheader("Top Crops Analysis")

        fig, ax = plt.subplots(figsize=(14,6))
        ax.bar(plot_df["Crop"], plot_df["Score"])
        ax.set_ylabel("Score")
        plt.xticks(rotation=90)

        st.pyplot(fig)

        # -----------------------------
        # RESULT
        # -----------------------------
        best_two = [scores[0][0], scores[1][0]]

        st.markdown("""
        <div style="padding:20px; background:#f0f2f6; border-radius:10px;">
        <h3 style='text-align:center;'>Recommended Crop Combination</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<h2 style='text-align:center; color:#2E7D32;'>{best_two[0]} + {best_two[1]}</h2>", unsafe_allow_html=True)
