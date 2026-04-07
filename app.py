import streamlit as st
import requests

st.title("FRAUD DETECTION SYSTEM")
st.write("Enter Transaction Detials")

step=st.selectbox("step",[1])
type=st.selectbox("type",["PAYMENT","TRANSFER","CASH_OUT","DEBIT"])
amount=st.number_input("Enter The Amount")
oldbalanceOrg=st.number_input("Enter The Old Balance")
newbalanceOrig=st.number_input("Enter The New Balance")
oldbalanceDest=st.number_input("Enter The Old Balance Dest")
newbalanceDest=st.number_input("Enter The New Balance Dest")

if st.button("Detect Transaction"):
    input_data={
        'step':step,
        'type':type,
        'amount':amount,
        'oldbalanceOrg':oldbalanceOrg,
        'newbalanceOrig':newbalanceOrig,
        'oldbalanceDest':oldbalanceDest,
        'newbalanceDest':newbalanceDest
    }
    response=requests.post("http://32.236.40.108:8001/predict",json=input_data)
    result=response.json()
    st.subheader("prediction result")
    st.success(result["prediction"])