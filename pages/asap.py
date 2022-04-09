from re import sub
import streamlit as st
import requests
from config import API_URL
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def app():
    st.title("Text Scoring Engine")

    st.markdown("""
                Here, you can evaluate the scoring framework on the ASAP-AES scoring dataset. Select an essay prompt
                below and the system will return the predicted score.
                """)

    essay_set = st.radio("Select the essay set", ["Set 3", "Set 4", "Set 5", "Set 6"])

    response = st.text_area("Enter the relevant segment of the report here.")

    submit = st.button("Submit")

    if submit:
        with st.spinner('Working'):
            score = requests.post(f"{API_URL}/predict-asap-aes/", json={"text": response, "essay_set": "".join(essay_set.lower().split())}).json()
            st.success(f"Your score is: {str(score['eval_score'])} / 4")
            chart = st.progress((score['eval_score']/4) * 1)
            if score['eval_score'] >= 3:
                st.balloons()

        fig, ax = plt.subplots()
        attentions = requests.get(API_URL + '/word-level-attention').json()
        print(attentions)

        key_phrase = st.selectbox("Select a key phrase to view attention heatmap", score["keywords"])

        key_phrase_index = score["keywords"].index(key_phrase)
        # fig = px.imshow(np.array(attentions[0][key_phrase_index])[:len(key_phrase.split()),:len(score['text'].split())], width=500, height=2000)
        fig = go.Figure(data=go.Heatmap(z=np.array(attentions[0][key_phrase_index])[:len(key_phrase.split()),:len(score['text'].split())], x=score['text'].split(), y=key_phrase.split()))
        # fig.layout.height = 400
        # fig.layout.width = 600
        fig.update(layout_coloraxis_showscale=False)
        fig.layout.height = 600
        fig.layout.width = 800
        st.write(fig)
            # ax = sns.heatmap(attentions[0][0], cmap='viridis', xticklabels=False, yticklabels=False, ax=ax)
            # st.write(fig)