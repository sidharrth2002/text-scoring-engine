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
    st.markdown("""
                ### The Paper Abstract
                 Academic scoring is mainly explored through the pedagogi- cal fields of Automated Essay Scoring (AES) and Short Answer Scoring (SAS), but text scoring in other domains has received limited attention. This paper focuses on industrial text scoring, namely the processing and adherence checking of long annual reports based on regulatory require- ments. To lay the foundations for non-academic scoring, a pioneering corpus of annual reports from Malaysian listed companies is scraped, seg- mented into sections, and domain experts score relevant sections based on adherence. Subsequently, deep neural non-hierarchical attention-based LSTMs, hierarchical attention networks and longformer-based models are refined and evaluated. Since the longformer outperformed LSTM- based models, we embed it into a hybrid scoring framework that employs lexicon and named entity features, with rubric injection via word-level attention, culminating in a Kappa score of 0.956 and 0.811 in both our corpora, respectively. Though scoring is fundamentally subjective, our proposed models show significant results when navigating thin rubric boundaries and handling adversarial responses. As our work proposes a novel industrial text scoring engine, we hope to validate our framework using more official documentation based on a broader range of regula- tory practices, both in Malaysia and the securities commissions of other nations.
                """)
