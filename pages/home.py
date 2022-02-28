import streamlit as st

def app():
    st.title("Text Scoring Engine")
    st.markdown("""
                This document is the demonstration of various models built for content-based scoring.
                My final framework is a combination of the longformer, NLP features and word-level attention. However, it has
                evolved through:\n
                1. Non-hierarchical LSTM-CNN-Attention\n
                2. Non-hierarchical LSTM-CNN-Mean Over Time\n
                3. Hierarchical LSTM-CNN-Attention\n
                4. Longformer\n
                5. Longformer + NLP Features\n
                6. Longformer + NLP Features + Word-level Attention
                """)
    st.markdown("""
                By the time anyone sees this page, the projects will be open sourced:\n
                1. The Multimodal Deep Learning Framework Written in Pytorch (built on top of Ken Gu's implementation)\n
                2. The API server that hosts the models\n
                3. All training notebooks with hyperparameters
                """)
