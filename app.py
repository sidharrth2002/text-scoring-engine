from multiapp import MultiPage
from pages import asap, home
import streamlit as st

app = MultiPage()

# app.add_page("Home", home.app)
# app.add_page("ASAP", asap.app)

app.run()