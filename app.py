import streamlit as st
import pandas as pd

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
        st.error("No data available")
    else:
        # Average yield per crop
        crop_yield = filtered.groupby('Crop')['Yield'].mean().sort_values(ascending=False)

        # -----------------------------
        # GRAPH (REAL MEANING)
        # -----------------------------
        import matplotlib.pyplot as plt

# Average yield calculation
crop_yield = filtered.groupby('Crop')['Yield'].mean().sort_values(ascending=False)

# Plot graph
st.subheader("📊 Crop Yield (Average kg/ha)")

fig, ax = plt.subplots()

ax.bar(crop_yield.index, crop_yield.values)

ax.set_xlabel("Crops")
ax.set_ylabel("Average Yield (kg/ha)")   # ✅ THIS IS YOUR Y-AXIS LABEL
ax.set_title("Crop Yield Comparison")

plt.xticks(rotation=90)

st.pyplot(fig)

        # -----------------------------
        # BEST COMBINATION
        # -----------------------------
        if len(crop_yield) < 2:
            st.error("Not enough crops for combination")
        else:
            best_two = crop_yield.head(2).index.tolist()

            st.success("🌾 Best Crop Combination")

            st.markdown(f"## 👉 {best_two[0]} + {best_two[1]}")

            # -----------------------------
            # OPTIONAL (simple clarity)
            # -----------------------------
            st.write("🌱 Based on highest average yield in selected region & season")
