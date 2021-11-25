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
from custom_layers.attention import Attention
from custom_layers.zero_masked_entries import ZeroMaskedEntries

# EMBEDDING_DIM = 300
# MAX_WORDS = 6000
# MAX_SEQUENCE_LENGTH = 700
# VALIDATION_SPLIT = 0.2
# DELTA = 20
# vocab_size = 3467

# with open('./models/A_tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# embedding_matrix = np.load('./models/A_embedding_matrix.npy')

# def practice_A_model():
#     input_layer = Input(name='response', shape=(MAX_SEQUENCE_LENGTH,))

#     embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], mask_zero=True, trainable=False)
#     word_embeddings = embedding_layer(input_layer)
#     word_embeddings_masked = ZeroMaskedEntries(name='pos_x_maskedout')(word_embeddings)

#     first_convolution = Conv1D(50, 3, padding='valid')(word_embeddings_masked)

#     first_lstm_layer = LSTM(300, return_sequences=True, recurrent_dropout=0.4, dropout=0.4)(first_convolution)
#     first_dropout = Dropout(0.3)(first_lstm_layer)

#     lstm_means = Attention()(first_dropout)
#     embedding_dense = Dense(64)(lstm_means)

#     applied_input = Input(name='applied', shape=(1,))
#     applied_dense = Dense(16)(applied_input)

#     concatenated = Concatenate()([embedding_dense, applied_dense])
#     second_dropout = Dropout(0.3)(concatenated)
#     score_dense = Dense(32)(second_dropout)
#     score_final = Dense(5, activation='softmax', name='score')(score_dense)

#     inputs = {'response': input_layer, 'whether_criteria_applied': applied_input}
#     outputs = {'score': score_final}

#     model = Model(inputs=inputs, outputs=outputs, name='Taghipour')
#     model.emb_index = 0

#     loss = {'score': CategoricalCrossentropy(from_logits=True)}
#     metric = {'score': CategoricalAccuracy('accuracy')}

#     optimizer = Adam()

#     model.compile(loss=loss, optimizer=optimizer, metrics=metric)

#     model.summary()

#     return model

# A_model = practice_A_model()
# A_model.load_weights('./models/practice-A-model')

model3 = load_model('./models/practice-A-model', custom_objects={'Attention': Attention})