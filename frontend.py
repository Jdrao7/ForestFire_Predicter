import streamlit as st
import numpy as np 
import pickle 

st.title("🔥 Fire Prediction in Algerian Forest")

month = st.number_input("Month", min_value=1, max_value=12, value=7)
temperature = st.number_input("Temperature (°C)", value=35.0)
RH = st.number_input("Relative Humidity (%)", value=45)
Ws = st.number_input("Wind Speed (km/h)", value=15)
Rain = st.number_input("Rain (mm)", value=0.0)
FFMC = st.number_input("FFMC", value=90.2)
FWI = st.number_input("FWI", value=20.8)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if st.button("Predict"):
    input_data = np.array([[month, temperature, RH, Ws, Rain, FFMC, FWI]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🔥 Warning: Fire Predicted!")
    else:
        st.success("✅ No Fire Predicted")
