import pandas as pd
import numpy as np
import joblib
import streamlit as st

with open('rf_grid.pkl', 'rb') as file_1:
  model = joblib.load(file_1)

sample_val = ['lab 1', 'lab 2']

st.header('Plant Nutrition Prediction')
v1 = st.number_input('Input V1:', 227.285714,678.375000)
v2 = st.number_input('Input V2:', 178.800000,422.812500)
v3 = st.number_input('Input V3:', 348.933333,722.312500)
v4 = st.number_input('Input V4:', 313.733333,558.500000)
v5 = st.number_input('Input V5:', 373.333333,721.000000)
v6 = st.number_input('Input V6:', 189.200000,415.375000)
v7 = st.number_input('Input V7:', 586.266667,853.466667)
v8 = st.number_input('Input V8:', 3725.666667,5086.375000)
sample_type = st.radio('sample_type', (sample_val))


if st.button('Predict'):
    data_inf = pd.DataFrame({'v1' : v1, 'v2' : v2, 
                        'v3' : v3, 'v4' : v4, 
                        'v5' : v5, 'v6' : v6, 
                        'v7' : v7, 'v8' : v8, 'sample_type' : sample_type}, index = [0])
    
    cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8']

    for col in cols:
        data_inf[col] = np.log(1 + data_inf[col])

    result = model.predict(data_inf)
    st.subheader(result)