import streamlit as st
import pandas as pd
import numpy as np
@st.cache_data
def load_data():
    df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
    df = df.dropna(subset=['Global_Sales'])
    df['Sales_Category'] = np.where(df['Global_Sales'] >= 1.0, 'Hit', 'Flop')
    return df
df = load_data()
st.title("Video Game Hit/Flop Predictor")
st.markdown("This app predicts whether a game is a **Hit** (≥ 1M Global Sales) or a **Flop**.")
st.sidebar.header("Input Game Details")
name = st.sidebar.text_input("Game Name", "Your Game")
platform = st.sidebar.selectbox("Platform", df['Platform'].dropna().unique())
genre = st.sidebar.selectbox("Genre", df['Genre'].dropna().unique())
publisher = st.sidebar.selectbox("Publisher", df['Publisher'].dropna().unique())
year = st.sidebar.slider("Release Year", int(df['Year_of_Release'].min()), int(df['Year_of_Release'].max()), 2010)
platform_mean = df[df['Platform'] == platform]['Global_Sales'].mean()
genre_mean = df[df['Genre'] == genre]['Global_Sales'].mean()
predicted_sales = (platform_mean + genre_mean) / 2
if predicted_sales >= 1.0:
    result = "Hit Game!"
else:
    result = "Flop Game"
st.subheader("Prediction")
st.markdown(f"**Predicted Global Sales:** `{predicted_sales:.2f} million` units")
st.success(result)
with st.expander("See Similar Games"):
    st.dataframe(df[(df['Platform'] == platform) & (df['Genre'] == genre)][['Name', 'Platform', 'Genre', 'Global_Sales']])
