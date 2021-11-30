# Text Scoring Engine
*Sidharrth Nagappan*

### Abstract
While academic scoring is primarily explored through Automated Essay Scoring (AES), text scoring in other domains has received limited attention. The work of this thesis culminates in an industrial text scoring engine that conducts adherence checking on an organisations’ annual reports using official regulatory rubrics, to port over the same benefits of automation that scholarly institutions have enjoyed. To pave the way for non-academic scoring, a corpus is compiled from segments of Malaysian public listed annual reports, cleaned meticulously and labelled by domain experts. The subsequent exploratory analysis uncovered similar TF-IDF and NER patterns between responses of the same score, as a precursor for feature engineering. An ablation study of neural hyperparameters was conducted and a series of multi-task, multi-channel and hierar- chical attention-based LSTM deep neural networks were tuned and evaluated, while transformer architectures (BERT and BigBird) and hybrid frameworks will be tested over the next phase. Preliminary results show that hierarchical networks are most ef- fective for long document scoring (Kappa of 0.778), while non-hierarchical LSTMs report better performance for relatively shorter text (Kappa of 0.729). Although the subjectivity of automated scoring is apparent, these scoring models show remarkable results when navigating thin boundaries, such that another look at the ground truth la- bel could lead a human scorer to rethink their original decision. Models often fail to flag adversarial responses, but sufficient corpus augmentation with incoherent text and the injection of handcrafted features are a plausible direction for FYP2.

### Instructions

Run the following command to install all dependencies:
```
pip install -r requirements.txt
```

Then run the app:
```
streamlit run app.py
```