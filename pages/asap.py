from re import sub
import streamlit as st
import requests
from config import API_URL

def app():
    st.title("Text Scoring Engine")

    st.markdown("""
                Here, you can evaluate the scoring framework on the ASAP-AES scoring dataset. Select an essay prompt
                below and the system will return the predicted score.sdf
                """)
    
    essay_set = st.radio("Select the essay set", ["Set 3", "Set 4", "Set 5", "Set 6"])

    response = st.text_area("Enter the relevant segment of the report here.")


    submit = st.button("Submit")

    if submit:
        entities = requests.post(f"{API_URL}/named-entities/", json={"text": response}).json()
        features = requests.post(f"{API_URL}/asap-features/", json={"text": response}).json()
        score = requests.post(f"{API_URL}/predict-asap-aes/", json={"text": response, "essay_set": "".join(essay_set.lower().split())}).json()

        st.write(entities)
        st.write(features)
        st.write(score)