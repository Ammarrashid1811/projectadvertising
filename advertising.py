import streamlit as st
import pandas as pd
import pickle

st.write("""
# Sales Prediction App

This app predicts the **Advertising Sales** based on TV, Newspaper and Radio expenses!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.70, 300.00, 150.00)
    Radio = st.sidebar.slider('Radio', 0, 50.00, 25.00)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 115.00, 70.00)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertisingmodel.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
