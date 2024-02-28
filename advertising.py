import streamlit as st
import pandas as pd
import pickle

st.write("""
# Sales Prediction App

This app predicts the **Advertising Sales** based on TV, Newspaper and Radio expenses!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    tv = st.sidebar.slider('TV', 0.70, 300.0, 150.0)
    radio = st.sidebar.slider('Radio', 0, 50.0, 25.0)
    newspaper = st.sidebar.slider('Newspaper', 0.30, 115.0, 70.0)
    data = {'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertisingmodel.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
