from multiapp import MultiPage
from pages import asap, home
import streamlit as st

st.set_page_config(page_title="Text Scoring Engine", page_icon="🚀")

app = MultiPage()

app.add_page("ASAP", asap.app)
app.add_page("Brief", home.app)

app.run()