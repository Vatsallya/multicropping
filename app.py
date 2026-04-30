import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df.columns = df.columns.str.strip()

    # Fix column names if different
    if 'State' in df.columns:
        df.rename(columns={'State': 'State_Name'}, inplace=True)

    # Create Yield if missing
    if 'Yield' not in df.columns:
        if 'Production' in df.columns and 'Area' in df.columns:
            df['Yield'] = df['Production'] / df['Area']
        else:
            st.error("Dataset must contain Yield or (Production & Area)")
            st.stop()

    # Normalize text
    df['State_Name'] = df['State_Name'].astype(str).str.lower().str.strip()
    df['Season'] = df['Season'].astype(str).str.lower().str.strip()
    df['Crop'] = df['Crop'].astype(str).str.strip()

    return df

df = load_data()

# -----------------------------
# ENCODING + MODEL
# -----------------------------
@st.cache_resource
def train_model(df):
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    df['State_enc'] = le_state.fit_transform(df['State_Name'])
    df['Season_enc'] = le_season.fit_transform(df['Season'])
    df['Crop_enc'] = le_crop.fit_transform(df['Crop'])

    X = df[['State_enc', 'Season_enc', 'Yield']]
    y = df['Crop_enc']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_state, le_season, le_crop

model, le_state, le_season, le_crop = train_model(df)

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

crop_list = sorted(filtered['Crop'].unique())
st.write(", ".join(crop_list))

# -----------------------------
# YIELD CALCULATION (kg/ha)
# -----------------------------
crop_yield = filtered.groupby('Crop')['Yield'].mean()

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

    avg_yield = filtered['Yield'].mean()

    input_data = [[
        le_state.transform([state])[0],
        le_season.transform([season])[0],
        avg_yield
    ]]

    probs = model.predict_proba(input_data)[0]

    # Normalize yield
    yield_norm = crop_yield / crop_yield.max()

    scores = []

    for crop in crop_yield.index:
        try:
            idx = le_crop.transform([crop])[0]
            model_score = probs[idx]
            yield_score = yield_norm[crop]

            # HYBRID SCORE
            final_score = (0.6 * model_score) + (0.4 * yield_score)

            scores.append((crop, final_score))
        except:
            continue

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    if len(scores) < 2:
        st.error("Not enough crops for combination")
        st.stop()

    best_crop1 = scores[0][0]
    best_crop2 = scores[1][0]

    st.success(f"👉 {best_crop1} + {best_crop2}")
