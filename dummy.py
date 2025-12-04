import streamlit as st

# Title
st.title("Simple Addition App")

# Inputs
num1 = st.number_input("Enter first number", value=0.0)
num2 = st.number_input("Enter second number", value=0.0)

# Button
if st.button("Add"):
    result = num1 + num2
    st.success(f"Result: {result}")
