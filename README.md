🌾 Smart Crop Recommendation System

An AI-powered crop recommendation web application that suggests the most suitable crops based on state, season, and geo-spatial cropping patterns using machine learning models.

🚀 Overview

This project combines:

📊 Crop production dataset
🌍 Geo-spatial NetCDF data
🤖 Machine Learning models

to build an intelligent system that helps in data-driven agricultural decision-making.

🎯 Objectives
Predict the best crops for a given region
Analyze cropping intensity across India
Integrate geo-spatial data with ML models
Provide an interactive web interface
🧠 Methodology
1. Data Collection
Crop Production Dataset (India)
NetCDF files (multi-cropping levels)
2. Data Processing
Cleaned and normalized dataset
Extracted:
level1 → single cropping
level2 → double cropping
level3 → triple cropping
3. Geo-Mapping
Used latitude & longitude filtering
Mapped cropping data to Indian regions
4. Feature Engineering

Created features:

State (encoded)
Season (encoded)
Cropping levels (level1, level2, level3)
Total cropping
Yield
5. Model Building

Models used:

🌲 Decision Tree (baseline)
🌳 Random Forest
⚡ XGBoost
6. Evaluation
Metric	Value
Top-1 Accuracy	~0.42
Top-3 Accuracy	~0.73
