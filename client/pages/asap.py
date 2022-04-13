from re import sub
import streamlit as st
import requests
from config import API_URL
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from rubrics import report_rubrics

def app():

    if "scored" not in st.session_state:
        st.session_state.scored = False
    if "options" not in st.session_state:
        st.session_state.options = False
    if st.session_state.get('step') is None:
        st.session_state['step'] = 0

    st.title("Industrial Text Scoring Engine")

    st.markdown("""
        Here, you can evaluate the scoring framework on either the ASAP-AES scoring dataset. Select an essay prompt
        below and the system will return the predicted score. Note that the training set is not yet fully extensive and this system
        is not yet production-ready.
    """)

    text_input_form = st.form(key='text-input-form')
    essay_set = text_input_form.radio("Select the essay set", [
                            "Set 3", "Set 4", "Set 5", "Set 6", "Practice A", "Practice B"])

    response = text_input_form.text_area("Enter the relevant text here. You can use Grammarly to help you out.")

    def form_callback_scored():
        st.session_state.scored = True

    def form_callback_options():
        st.session_state.options = True

    submit = text_input_form.form_submit_button(label='Go!', on_click=form_callback_scored)

    if submit or st.session_state.get('step') == 1:
        if 'Set' in essay_set:
            # using ASAP-AES dataset
            with st.spinner('Working'):
                score = requests.post(f"{API_URL}/predict-asap-aes/", json={
                                      "text": response, "essay_set": "".join(essay_set.lower().split())}).json()
                st.success(f"Your score is: {str(score['eval_score'])} / 4")
                chart = st.progress((score['eval_score']/4) * 1)
                if score['eval_score'] >= 3:
                    st.balloons()
        elif 'Practice' in essay_set:
            # using annual reports
            with st.spinner('Working'):
                if st.session_state.get('step') == 0:
                    score = requests.post(f"{API_URL}/predict-report/", json={
                                        "text": response, "essay_set": "-".join(essay_set.lower().split())}).json()
                    st.session_state['score'] = score
                    col1, col2 = st.columns(2)
                    col1.metric("Score", score['eval_score'])
                    col2.markdown(report_rubrics[score['eval_score']])
                    st.session_state['step'] = 1
                # st.success(f"Your score is: {str(score['eval_score'])} / 4")
                # chart = st.progress((score['eval_score']/4) * 1)

        score = st.session_state.get('score')
        keywords = score['keywords']
        fig, ax = plt.subplots()
        attentions = requests.get(API_URL + '/word-level-attention').json()

        attention_form = st.form(key='attention-form')

        key_phrase = attention_form.selectbox("Select a key phrase to view attention heatmap", keywords, key="select_keyword")
        form_submit = attention_form.form_submit_button("View Heatmap", on_click=form_callback_options)

        if form_submit:
            st.write(keywords.index(key_phrase))
            key_phrase_index = keywords.index(key_phrase)
            # fig = px.imshow(np.array(attentions[0][key_phrase_index])[:len(key_phrase.split()),:len(score['text'].split())], width=500, height=2000)
            fig = go.Figure(data=go.Heatmap(z=np.array(attentions[0][key_phrase_index])[:len(
                key_phrase.split()), :len(score['text'].split())], x=score['text'].split(), y=key_phrase.split()))
            # fig.layout.height = 400
            # fig.layout.width = 600
            fig.update(layout_coloraxis_showscale=False)
            fig.layout.height = 600
            fig.layout.width = 800
            st.write(fig)
    # ax = sns.heatmap(attentions[0][0], cmap='viridis', xticklabels=False, yticklabels=False, ax=ax)
    # st.write(fig)
