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

    # Normalize
    df.columns = df.columns.str.strip()
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

    features = ['State_enc','Season_enc','level1','level2','level3','total','Yield']

    for col in features:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    X = df[features]
    y = df['Crop_enc']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_state, le_season, le_crop

model, le_state, le_season, le_crop = train_model(df)

# -----------------------------
# UI
# -----------------------------
st.title("Smart Crop Recommendation")

state = st.selectbox("State", sorted(df['State_Name'].unique()))
season = st.selectbox("Season", sorted(df['Season'].unique()))

filtered = df[
    (df['State_Name'] == state) &
    (df['Season'] == season)
]

st.subheader("Available Crops")

if filtered.empty:
    st.warning("No data found")
else:
    st.write(", ".join(filtered['Crop'].unique()))

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Recommend"):

    if filtered.empty:
        st.error("No data")
        st.stop()

    try:
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

        crop_yield = filtered.groupby('Crop')['Yield'].mean()
        yield_norm = crop_yield / crop_yield.max()

        scores = []

        for crop in crop_yield.index:
            try:
                idx = le_crop.transform([crop])[0]
                model_prob = probs[idx]
                yield_score = yield_norm[crop]

                final_score = (0.6 * model_prob) + (0.4 * yield_score)

                scores.append((crop, final_score))
            except:
                continue  # skip problematic crops

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        if len(scores) < 2:
            st.error("Not enough crops")
            st.stop()

        # GRAPH
        plot_df = pd.DataFrame(scores[:10], columns=["Crop", "Score"])

        fig, ax = plt.subplots(figsize=(12,5))
        ax.bar(plot_df["Crop"], plot_df["Score"])
        ax.set_xlabel("Crops")
        ax.set_ylabel("Score")
        plt.xticks(rotation=90)

        st.pyplot(fig)

        # BEST COMBINATION
        st.success(f"{scores[0][0]} + {scores[1][0]}")

    except Exception as e:
        st.error(f"Error: {e}")
