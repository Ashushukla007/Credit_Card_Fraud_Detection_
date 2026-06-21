import streamlit as st
import requests

st.title("Fraud Detection Model")
st.markdown("APP IS RUNNING")

row=st.selectbox("select the row between 0 and 982", range(983))

if st.button("Predict"):
    response=requests.post("http://127.0.0.1:8000/prediction?row_number=" + str(row))
    result=response.json()
    
    if result['prediction']==1:
        st.error("Fraudulent Transaction")
    else:
        st.success("Legitimate Transaction")
    
    st.metric(
        "Fraud Probability",
        f"{result['probability']:.2%}"
    )