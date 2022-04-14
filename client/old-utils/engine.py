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
