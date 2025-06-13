import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
reference_data = pd.read_csv("reference_encoded_data.csv")
reference_columns = reference_data.drop(columns=["Global_Sales", "Sales_Class"]).columns
st.title("Video Game Hit/Flop Predictor")
st.markdown("This app predicts whether a game is a **Hit** (≥ 1M Global Sales) or a **Flop**.")
st.sidebar.header("Input Game Details")
game_name = st.sidebar.text_input("Game Name", "Mario Kart")
platform = st.sidebar.selectbox("Platform", reference_data['Platform'].unique())
genre = st.sidebar.selectbox("Genre", reference_data['Genre'].unique())
publisher = st.sidebar.selectbox("Publisher", reference_data['Publisher'].unique())
year = st.sidebar.slider("Release Year", 1980, 2020, 2008)
input_df = pd.DataFrame({
    "Name": [game_name],
    "Platform": [platform],
    "Genre": [genre],
    "Publisher": [publisher],
    "Year_of_Release": [year]
})
input_encoded = pd.get_dummies(input_df.drop(columns=["Name"]))
for col in reference_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[reference_columns]
num_cols = ['Year_of_Release']
input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
predicted_sales = model.predict(input_encoded)[0]
predicted_sales_exp = np.expm1(predicted_sales)
st.write("Raw model prediction:", predicted_sales_exp)
if predicted_sales_exp >= 1.0:
    st.success(f"**Hit Game**! Predicted Global Sales: **{predicted_sales_exp:.2f} million** units")
else:
    st.error(f"**Flop Game**. Predicted Global Sales: **{predicted_sales_exp:.2f} million** units")
similar_games = reference_data[(reference_data['Platform'] == platform) & 
                               (reference_data['Genre'] == genre)].copy()
similar_games = similar_games.sort_values("Global_Sales", ascending=False)[["Name", "Platform", "Genre", "Global_Sales"]]
st.markdown("### See Similar Games")
st.dataframe(similar_games.head(10))
