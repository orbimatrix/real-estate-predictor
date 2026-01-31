import streamlit as st
import requests

st.title("🏡 Real Estate Price Predictor")
st.write("Enter house details below to get a price prediction.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("Crime Rate (CRIM)", value=0.006)
    zn = st.number_input("Residential Land Zone (ZN)", value=18.0)
    indus = st.number_input("Non-retail Business (INDUS)", value=2.31)
    chas = st.selectbox("Near River? (CHAS)", options=[0, 1])
    nox = st.number_input("Nitric Oxides (NOX)", value=0.538)
    rm = st.number_input("Average Rooms (RM)", value=6.5)

with col2:
    age = st.number_input("Built before 1940 (AGE)", value=65.2)
    dis = st.number_input("Distance to Centers (DIS)", value=4.09)
    rad = st.number_input("Highway Accessibility (RAD)", value=1)
    tax = st.number_input("Property Tax (TAX)", value=296)
    ptratio = st.number_input("Pupil-Teacher Ratio", value=15.3)
    b = st.number_input("B Index", value=396.9)
    lstat = st.number_input("% Lower Status (LSTAT)", value=4.98)

if st.button("Predict Price"):
    # Prepare payload for FastAPI
    payload = {
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas,
        "NOX": nox, "RM": rm, "AGE": age, "DIS": dis,
        "RAD": rad, "TAX": tax, "PTRATIO": ptratio, "B": b, "LSTAT": lstat
    }
    
    # Call your FastAPI endpoint
    response = requests.post("http://localhost:8000/predict", json=payload)
    
    if response.status_code == 200:
        prediction = response.json()["predicted_price_k"]
        st.success(f"### Predicted Price: ${prediction}k")
    else:
        st.error("Error in prediction. Check if FastAPI is running.")