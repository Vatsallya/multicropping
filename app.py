import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Smart Crop Recommendation", layout="centered")

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
# TRAIN MODEL
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

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)

    return model, le_state, le_season, le_crop

model, le_state, le_season, le_crop = train_model(df)

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

# -----------------------------
# AVAILABLE CROPS
# -----------------------------
st.subheader("🌱 Available Crops")

if filtered.empty:
    st.warning("No data available")
else:
    st.write(", ".join(sorted(filtered['Crop'].unique())))

# -----------------------------
# RECOMMENDATION
# -----------------------------
if st.button("Recommend Best Combination"):

    if filtered.empty:
        st.error("No data for prediction")
    else:
        # Average feature values
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
        # SCORE CALCULATION (FIXED)
        # -----------------------------
        crop_stats = filtered.groupby('Crop').agg({
            'Yield': 'mean'
        }).reset_index()

        crop_stats['Yield_norm'] = crop_stats['Yield'] / crop_stats['Yield'].max()

        scores = []

        for _, row in crop_stats.iterrows():
            crop = row['Crop']

            if crop not in le_crop.classes_:
                continue

            idx = le_crop.transform([crop])[0]
            prob = probs[idx]

            # Ignore very weak predictions
            if prob < 0.01:
                continue

            final_score = (0.5 * prob) + (0.5 * row['Yield_norm'])

            scores.append((crop, prob, row['Yield_norm'], final_score))

        if len(scores) < 2:
            st.error("Not enough strong crops for recommendation")
            st.stop()

        # Sort by final score
        scores = sorted(scores, key=lambda x: x[3], reverse=True)

        # -----------------------------
        # VISUALIZATION (CONSISTENT)
        # -----------------------------
        top_scores = scores[:10]

        plot_df = pd.DataFrame({
            "Crop": [x[0] for x in top_scores],
            "Score": [x[3] for x in top_scores]
        })

        st.subheader("📊 Top Crops (Final Score)")
        st.bar_chart(plot_df.set_index("Crop"))

        # -----------------------------
        # BEST COMBINATION (CORRECT)
        # -----------------------------
        crop1, p1, y1, s1 = scores[0]
        crop2, p2, y2, s2 = scores[1]

        combined = round(((s1 + s2) / 2) * 100, 2)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.success("🌾 Best Crop Combination")

        st.markdown(f"## 👉 {crop1} + {crop2}")

        st.subheader("📊 Confidence Analysis")

        st.write(f"{crop1} → Model: {round(p1*100,2)}% | Yield: {round(y1*100,2)}% | Final Score: {round(s1*100,2)}%")
        st.write(f"{crop2} → Model: {round(p2*100,2)}% | Yield: {round(y2*100,2)}% | Final Score: {round(s2*100,2)}%")

        st.write(f"🌾 Combined Score → {combined}%")

        st.progress(float(s1))
        st.progress(float(s2))

        st.info("This is a data-driven recommendation based on historical yield and model prediction, not a guaranteed real-world outcome.")
