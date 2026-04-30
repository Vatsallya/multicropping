import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from itertools import combinations

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

st.subheader("🌱 Available Crops")

if filtered.empty:
    st.warning("No data available")
else:
    crops = sorted(filtered['Crop'].unique())
    st.write(", ".join(crops))

# -----------------------------
# SHOW REAL COMBINATIONS
# -----------------------------
st.subheader("🔗 Common Crop Combinations")

if not filtered.empty:
    combo_list = []

    # Create combinations from dataset rows
    crop_counts = filtered['Crop'].value_counts()

    top_crops = crop_counts.head(10).index.tolist()

    for c1, c2 in combinations(top_crops, 2):
        combo_list.append((f"{c1} + {c2}", crop_counts[c1] + crop_counts[c2]))

    combo_df = pd.DataFrame(combo_list, columns=["Combination", "Score"])
    combo_df = combo_df.sort_values(by="Score", ascending=False).head(10)

    st.bar_chart(combo_df.set_index("Combination"))

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

        # -----------------------------
        # HYBRID SCORING
        # -----------------------------
        crop_stats = filtered.groupby('Crop').agg({
            'Yield': 'mean'
        }).reset_index()

        crop_stats['Yield_norm'] = crop_stats['Yield'] / crop_stats['Yield'].max()

        available = crop_stats['Crop'].values
        available_idx = le_crop.transform(available)

        crop_scores = []

        for crop, idx in zip(available, available_idx):
            prob = probs[idx]
            yield_score = crop_stats[crop_stats['Crop'] == crop]['Yield_norm'].values[0]

            final_score = (0.6 * prob) + (0.4 * yield_score)

            crop_scores.append((crop, prob, yield_score, final_score))

        crop_scores = sorted(crop_scores, key=lambda x: x[3], reverse=True)

        # BEST 2 CROPS
        crop1, p1, y1, s1 = crop_scores[0]
        crop2, p2, y2, s2 = crop_scores[1]

        # Combined (more realistic)
        combined = round((s1 * s2) * 100, 2)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.success("🌾 Best Practical Crop Combination")

        st.markdown(f"## 👉 {crop1} + {crop2}")

        st.subheader("📊 Confidence Analysis")

        st.write(f"🌱 {crop1} → Model: {round(p1*100,2)}% | Yield Score: {round(y1*100,2)}%")
        st.write(f"🌱 {crop2} → Model: {round(p2*100,2)}% | Yield Score: {round(y2*100,2)}%")

        st.write(f"🌾 Combined Practical Score → {combined}%")

        # Progress bars
        st.progress(float(p1))
        st.progress(float(p2))

        st.info("⚠️ This is a data-driven recommendation based on historical patterns, not a guaranteed real-world outcome.")
        st.progress(p2)
