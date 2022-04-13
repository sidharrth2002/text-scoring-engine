'''
Sidharrth Nagappan
'''

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Bidirectional, Embedding, LSTM, Dense, Flatten, Dropout, concatenate, GlobalAveragePooling1D, Conv1D, TimeDistributed, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
import nltk
from custom_layers.attention import Attention
from custom_layers.zero_masked_entries import ZeroMaskedEntries

EMBEDDING_DIM = 300
MAX_WORDS = 6000
A_MAX_SEQUENCE_LENGTH = 700

VALIDATION_SPLIT = 0.2
DELTA = 20
vocab_size = 3467

with open('./models/A_tokenizer.pickle', 'rb') as handle:
    A_tokenizer = pickle.load(handle)

A_embedding_matrix = np.load('./models/A_embedding_matrix.npy')

with open('./models/B_tokenizer.pickle', 'rb') as handle:
    B_tokenizer = pickle.load(handle)
    print(B_tokenizer.word_index['independent'])

A_model = load_model('./models/practice-A-model', custom_objects={'Attention': Attention})
B_model = load_model('./models/practice-B-model')

def practice_A_model(response, applied):
    print(response)
    test_sequence = A_tokenizer.texts_to_sequences([response])
    test_sequence = pad_sequences(test_sequence, maxlen=A_MAX_SEQUENCE_LENGTH)
    return A_model.predict({'response': test_sequence, 'whether_criteria_applied': pd.Series([applied])})['score'].argmax(axis=1)[0]

def vectorise(response):
    max_features = 200000
    max_senten_len = 100
    max_senten_num = 35
    paras = []
    sentences = []
    texts = []
    texts.append(response)
    sentences = nltk.tokenize.sent_tokenize(response)
    sentences = [sent for sent in sentences if len(sent) > 2]
    paras.append(sentences)

    data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
    for i, sentences in enumerate(paras):
        for j, sent in enumerate(sentences):
            if j < max_senten_num:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_senten_len and B_tokenizer.word_index[word] < max_features:
                            data[i, j, k] = B_tokenizer.word_index[word]
                            k = k + 1
                    except:
                        pass
    return data

def practice_B_model(response, applied):
    print(response)
    test_sequence = vectorise(response)
    return B_model.predict(x={'response': test_sequence, 'whether_criteria_applied': pd.Series([applied])}).argmax(axis=1)[0]

# print(practice_B_model('The Board with assistance from the Nominating Committee NC undertakes to carry out a formal and objective annual evaluation to assess the performance and effectiveness of the Board and Board Committees, as well as the performance of each Director and each Audit Committee member. Each Director evaluates the performance of the Board and conducts a peer assessment of the other Directors. Each Board Committee member evaluates their respective Board Committee, while each Audit Committee member conducts a peer assessment of the other Audit Committee members. Upon completion of the evaluation form by each Director and Board Committee member, they shall submit their assessment to the Secretary of the NC, who will summarise the findings for submission to the NC. The NC will subsequently evaluate the assessment prior to its reporting and presentation to the Board. The NC also assesses the independence of Directors annually and focuses beyond the Independent Director s background, economic and family relationships to consider whether the Independent Director can continue to bring independent and objective judgment to Board deliberations. Based on the criteria specified in the Malaysian Code on Corporate Governance and the Main Market Listing Requirements of Bursa Malaysia Securities Berhad MMLR , a Director is considered independent if he/she: has fulfilled the criteria under the definition of Independent Director pursuant to the MMLR; has ensured effective check and balance in the proceedings of the Board and the Board Committees; has actively participated in the Board deliberations, provided objectivity in decision making and an independent voice to the Board; has consistently challenged Management in an effective and constructive manner; has kept a distance from Management in overseeing and monitoring execution of strategy; has not been engaged by the Company as an adviser under such circumstances as prescribed by the Bursa Malaysia Securities Berhad Bursa Securities or is not presently a Director except as Independent Director or major shareholder of a firm or corporation which provides professional advisory services to the Company under such circumstances as prescribed by the Bursa Securities; has not engaged in any transaction with the Company including transaction of assets and services, joint ventures, financial assistance etc. under such circumstances as prescribed by the Exchange or is not presently a Director except as Independent Director or major shareholder of a firm or corporation which has been engaged in any transaction with the Company under such circumstances as prescribed by the Bursa Securities; has not received any performance based remuneration or share based incentives from the Company, its subsidiaries, holding company or any of its related corporations; and has no other material relationship with the Company, either directly or as a partner, shareholder, director or officer of an organisation that has a material relationship with the Company. The NC upon its annual assessment carried out for financial year , was satisfied that: the size and composition of the Board is optimum with an appropriate mix of knowledge, skills, attributes and core competencies; the Board has been able to discharge its duties professionally and effectively in consideration of the scale and breadth of the Company s operations; all the Directors continue to uphold the highest governance standards in their conduct and that of the Board; all the members of the Board are well qualified to hold their positions as Directors of the Company in view of their respective depth of knowledge, skills and experience and their personal qualities; the Independent Non Executive Directors comply with the definition of Independent Director as defined in the MMLR; and the Directors are able to devote sufficient time commitment to their roles and responsibilities as Directors of the Company as reflected by their attendance at the Board meetings and Board Committee meetings. The Board noted the recommendation for Large Companies to engage independent experts periodically to facilitate objective and candid board evaluations. The Board in May has analysed the annual assessment carried out for financial year and was satisfied with the assessment result as mentioned above. Hence, the Board agreed that there is no necessity for QL to engage independent expert for the said exercise.', 1))