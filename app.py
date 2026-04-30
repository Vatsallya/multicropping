import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center;'>Smart Crop Recommendation System</h1>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# INPUT SECTION
# -----------------------------
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

# -----------------------------
# AVAILABLE CROPS
# -----------------------------
st.subheader("Available Crops")

if filtered.empty:
    st.warning("No data available for selected inputs")
else:
    st.write(", ".join(sorted(filtered['Crop'].unique())))

# -----------------------------
# BUTTON
# -----------------------------
if st.button("Recommend Best Combination"):

    if filtered.empty:
        st.error("No data available")

    else:
        crop_yield = filtered.groupby('Crop')['Yield'].mean().sort_values(ascending=False)

        # -----------------------------
        # GRAPH
        # -----------------------------
        

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(crop_yield.index, crop_yield.values)

        ax.set_xlabel("Crops")
        ax.set_ylabel("Average Yield (kg/ha)")
        ax.set_title("Yield Comparison")

        plt.xticks(rotation=90)

        st.pyplot(fig)

        # -----------------------------
        # RESULT CARD
        # -----------------------------
        if len(crop_yield) < 2:
            st.error("Not enough crops")
        else:
            best_two = crop_yield.head(2).index.tolist()

            st.markdown("""
            <div style="
                padding:20px;
                border-radius:10px;
                background-color:#f0f2f6;
                margin-top:20px;
            ">
                <h3 style='text-align:center;'>Recommended Crop Combination</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <h2 style='text-align:center; color:#2E7D32;'>
            {best_two[0]} + {best_two[1]}
            </h2>
            """, unsafe_allow_html=True)
