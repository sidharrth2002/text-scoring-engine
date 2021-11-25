import streamlit as st
import pandas as pd
from utils import encode
from engine import practice_A_model, practice_B_model

st.title("An Industrial Text Scoring Engine")
st.markdown("Pass in the relevant segment of your annual report and choose the practice you wish to check compliance for.")

st.markdown("**Note:** This is a prototype, so please be patient while the engine is being developed.")

st.markdown("### Practices")
st.markdown("A: Your company **appoints directors** to the board through **fair** and **transparent** means. You use **independent sources** like public directories and do not only rely on recommendations from existing board members.")
st.markdown("B: You **evaluate** your board of directors annually, and these **evaluations are independent**. Afterwhich, you have outlined the **relevant outcomes** and taken action as necessary.")

st.markdown("### Engine Demo")

practice = st.radio("Choose your practice", ["A", "B"])
applied = st.radio("Has the practice been applied? If not, justify why below", ["Yes", "No"])
response = st.text_area("Enter the relevant segment of the report here.")

submit = st.button("Submit")

if submit:
    if practice == "A":
        score = practice_A_model(response, encode(applied))
        st.success("Your adherence score is " + str(score))
    elif practice == "B":
        score = practice_B_model(response, encode(applied))
        st.success("Your adherence score is " + str(score))