from functools import partial
import logging
from os.path import join, exists

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from app.controllers.feature_generation import lemmatize
from app.dataclass.arguments import ModelArguments

from .tabular_torch_dataset import TorchTabularTextDataset
from .data_utils import (
    CategoricalFeatures,
    agg_text_columns_func,
    convert_to_func,
    get_matching_cols,
    load_num_feats,
    load_cat_and_num_feats,
    normalize_numerical_feats,
)

from torchtext.data import get_tokenizer
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from itertools import cycle

from transformers import AutoTokenizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

CATEGORICAL_ENCODE_TYPE = 'ohe'
EMPTY_TEXT_VALUES = ['nan', 'None']
REPLACE_EMPTY_TEXT = None
SEP_TEXT_TOKEN_STR = ' '

def get_sequence_length(essay_set):
    if essay_set=='set3':
        return 150
    elif essay_set=='set4':
        return 150
    elif essay_set=='set5':
        return 512
    elif essay_set=='set6':
        return 300

def get_token_num_for_keywords(group):
    if group == 'set3':
        return 15
    elif group == 'set4':
        return 15
    elif group == 'set5':
        return 13
    elif group == 'set6':
        return 10
    elif group == 'practice-a':
        return 15
    elif group == 'practice-b':
        return 15

def load_data_from_folder():
    pass

def load_data_into_folds():
    pass

def process_single_text_asap(text, source, features, keywords=[], essay_set='set3'):
    all_data = pd.read_csv(f'app/data/asap-aes/training_set_rel3_features.tsv', sep='\t', encoding='ISO-8859-1', index_col=0)
    all_data['essay'] = all_data['essay'].apply(lambda x: x.replace(r'/[^\w,.:;\[\]()/\!@#$%^&*+{}<>=?~|" -]/g', ''))
    all_data['essay'] = all_data['essay'].apply(lambda x: x.replace(r'/\s+/g', ''))

    max_token_length = int(all_data[all_data['essay_set'] == int(essay_set[-1])]['num_words'].max())

    print('Max token length is ', max_token_length)

    # means we are using ASAP-AES and not Bursa
    if 'set' in essay_set:
        max_keyword_length = max([len(i.split()) for i in keywords])
        print('Max keyword length is ', max_keyword_length)

    model_args = ModelArguments(
        model_name_or_path='bert-base-uncased'
    )

    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=model_args.cache_dir,
        max_sequence_length=max_token_length
    )

    #TODO: use pretrained tokenizer
    glove_tokenizer = Tokenizer(num_words=10000)
    glove_tokenizer.fit_on_texts(all_data[all_data['essay_set'] == int(essay_set[-1])]['essay'].to_list() + keywords)
    # glove_tokenizer.fit_on_texts(keywords + [text])

    print('Tokenizer vocab size is ', len(glove_tokenizer.word_index))

    print(features)
    if source == 'asap':
        categorical_cols = ['dummy_cat']
        text_cols = ['text']
        numerical_cols = list(features.keys())

        logger.info(f'Text columns: {text_cols}')
        logger.info(f'Categorical columns: {categorical_cols}')
        logger.info(f'Numerical columns: {numerical_cols}')

        text_cols_func = convert_to_func(text_cols)
        categorical_cols_func = convert_to_func(categorical_cols)
        numerical_cols_func = convert_to_func(numerical_cols)

        # calculate_features
        # features is a dictionary
        data_df = pd.DataFrame(features, index=[0])
        data_df['text'] = text
        data_df['lemmatized'] = ' '.join(lemmatize(text))
        # framework needs at least one categorical feature, but all ASAP-AES features are numerical
        data_df['dummy_cat'] = 1

        categorical_feats, numerical_feats = load_cat_and_num_feats(data_df,
                                                                    categorical_cols_func,
                                                                    numerical_cols_func,
                                                                    CATEGORICAL_ENCODE_TYPE)

        print('reach before')
        # there must be 1_ and 0_ for each cat feat, this is wrong
        # categorical_feats = np.array([np.concatenate((categorical_feats[0], categorical_feats[0]))])
        # if len(categorical_feats[0] == len(features['cat_cols'])):
        #     # it did not add 1 and 0 for each feature
        #     # new_cat_feats = np.array(categorical_feats[0])
        #     reversed_cat_feats = [int(not(i)) for i in categorical_feats[0]]
        #     categorical_feats = np.array([merge_lists_alternatively(categorical_feats[0], reversed_cat_feats)])

        print('reach after')

        numerical_feats = normalize_numerical_feats(numerical_feats, transformer=None)
        agg_func = partial(agg_text_columns_func, EMPTY_TEXT_VALUES, REPLACE_EMPTY_TEXT)
        texts_cols = get_matching_cols(data_df, text_cols_func)

        texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
        lemmatized_texts_list = data_df[['lemmatized']].agg(agg_func, axis=1).tolist()

        for i, text in enumerate(texts_list):
            texts_list[i] = f' {SEP_TEXT_TOKEN_STR} '.join(text)

        for i, text in enumerate(lemmatized_texts_list):
            lemmatized_texts_list[i] = f' {SEP_TEXT_TOKEN_STR} '.join(text)

        logger.info(f'Raw text example: {texts_list[0]}')
        logger.info(f'Lemmatized text example: {lemmatized_texts_list[0]}')

        hf_model_text_input = tokenizer(texts_list, padding="max_length", truncation=True,
                                        max_length=max_token_length)

        tokenized_text_ex = ' '.join(tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
        logger.debug(f'Tokenized text example: {tokenized_text_ex}')

        answer_tokens = glove_tokenizer.texts_to_sequences(texts_list)
        answer_tokens = pad_sequences(answer_tokens, maxlen=max_token_length, padding='post', truncating='post')

        answer_lemmatized_tokens = glove_tokenizer.texts_to_sequences(lemmatized_texts_list)
        answer_lemmatized_text = glove_tokenizer.sequences_to_texts(answer_lemmatized_tokens)
        # answer_lemmatized_tokens = [[i for i in j if i < len(glove_tokenizer.word_index)] for j in answer_lemmatized_tokens]
        answer_lemmatized_tokens = pad_sequences(answer_lemmatized_tokens, maxlen=max_token_length, padding='post', truncating='post')
        # create mask
        # change to lemmatized mask
        answer_mask = torch.zeros(answer_lemmatized_tokens.shape, dtype=torch.long)
        # print(torch.Tensor(answer_tokens))
        # FIXME: this is a hack to get the mask to work, I'm going to remove the part that makes it a tensor: answer_mask = torch.Tensor(answer_tokens)
        answer_mask.masked_fill_(torch.Tensor(answer_lemmatized_tokens) != 0, 1)

        keyword_tokens = glove_tokenizer.texts_to_sequences(keywords)
        keyword_tokens = pad_sequences(keyword_tokens, maxlen=max_keyword_length, padding='post', truncating='post')
        keyword_tokens = torch.reshape(torch.from_numpy(keyword_tokens), (len(keywords), max_keyword_length))
        keyword_mask = torch.zeros(keyword_tokens.shape, dtype=torch.long)
        keyword_mask.masked_fill_(keyword_tokens != 0, 1)

        return TorchTabularTextDataset(hf_model_text_input, categorical_feats,
                                    numerical_feats, answer_tokens, answer_mask, keyword_tokens, keyword_mask, [0, 1, 2, 3], data_df, label_list=None, class_weights=None,
                                    texts=texts_list,
                                    lemmatized_answer_tokens=answer_lemmatized_tokens, lemmatized_answer_texts=lemmatized_texts_list, answer_lemmatized_text=answer_lemmatized_text)

def merge_lists_alternatively(lst1, lst2):
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

def process_single_text_report(text, source, features, keywords=[], essay_set='practice-a'):
    practice = essay_set[-1]

    glove_tokenizer = Tokenizer(num_words=10000)
    if practice == 'a':
        train = pd.read_csv('app/data/a/train.csv')
        test = pd.read_csv('app/data/a/test.csv')
        val = pd.read_csv('app/data/a/val.csv')

        max_token_length = 700
        print('Max token length is ', max_token_length)

            #TODO: use pretrained tokenizer
        glove_tokenizer.fit_on_texts(train['lemmatized'].tolist() + val['lemmatized'].tolist() + test['lemmatized'].tolist() + keywords)
        # glove_tokenizer.fit_on_texts(keywords + [text])

    elif practice == 'b':
        train = pd.read_csv('app/data/b/train.csv', index_col=0)
        test = pd.read_csv('app/data/b/test.csv', index_col=0)
        print(train['lemmatized'])
        print(test['lemmatized'])
        max_token_length = 1024
        glove_tokenizer.fit_on_texts([str(i) for i in train['lemmatized'].tolist()] + [str(i) for i in test['lemmatized'].tolist()] + keywords)

    # means we are using ASAP-AES and not Bursa
    max_keyword_length = max([len(i.split()) for i in keywords])
    print('Max keyword length is ', max_keyword_length)

    model_args = ModelArguments(
        model_name_or_path='allenai/longformer-base-4096'
    )

    tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_or_name,
        cache_dir=model_args.cache_dir,
        max_sequence_length=max_token_length
    )

    print('Tokenizer vocab size is ', len(glove_tokenizer.word_index))

    print(features)
    categorical_cols = list(features['cat_cols'].keys())
    text_cols = ['text']
    numerical_cols = list(features['num_cols'].keys())

    logger.info(f'Text columns: {text_cols}')
    logger.info(f'Categorical columns: {categorical_cols}')
    logger.info(f'Numerical columns: {numerical_cols}')

    print('Categorical Feats', categorical_cols)
    print('Numerical Feats', numerical_cols)

    text_cols_func = convert_to_func(text_cols)
    categorical_cols_func = convert_to_func(categorical_cols)
    numerical_cols_func = convert_to_func(numerical_cols)

    # calculate_features
    # features is a dictionary
    # double the fucking features
    all_features = {**features['cat_cols'], **features['num_cols']}
    data_df = pd.DataFrame(all_features, index=[0])
    data_df['text'] = text
    data_df['lemmatized'] = ' '.join(lemmatize(text))
    data_df['lemmatized'] = data_df['lemmatized'].apply(remove_stopwords)
    # framework needs at least one categorical feature, but all ASAP-AES features are numerical

    categorical_feats, numerical_feats = load_cat_and_num_feats(data_df,
                                                                categorical_cols_func,
                                                                numerical_cols_func,
                                                                CATEGORICAL_ENCODE_TYPE)

    # there must be 1_ and 0_ for each cat feat, this is wrong
    # categorical_feats = np.array([np.concatenate((categorical_feats[0], categorical_feats[0]))])
    if len(categorical_feats[0] == len(features['cat_cols'])):
        # it did not add 1 and 0 for each feature
        # new_cat_feats = np.array(categorical_feats[0])
        reversed_cat_feats = [int(not(i)) for i in categorical_feats[0]]
        categorical_feats = np.array([merge_lists_alternatively(categorical_feats[0], reversed_cat_feats)])
    print('Categorical feats', len(categorical_feats))
    print(categorical_feats)

    numerical_feats = normalize_numerical_feats(numerical_feats, transformer=None)
    agg_func = partial(agg_text_columns_func, EMPTY_TEXT_VALUES, REPLACE_EMPTY_TEXT)
    texts_cols = get_matching_cols(data_df, text_cols_func)

    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    lemmatized_texts_list = data_df[['lemmatized']].agg(agg_func, axis=1).tolist()

    for i, text in enumerate(texts_list):
        texts_list[i] = f' {SEP_TEXT_TOKEN_STR} '.join(text)

    for i, text in enumerate(lemmatized_texts_list):
        lemmatized_texts_list[i] = f' {SEP_TEXT_TOKEN_STR} '.join(text)

    logger.info(f'Raw text example: {texts_list[0]}')
    logger.info(f'Lemmatized text example: {lemmatized_texts_list[0]}')

    hf_model_text_input = tokenizer(texts_list, padding="max_length", truncation=True,
                                    max_length=max_token_length)

    tokenized_text_ex = ' '.join(tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
    logger.debug(f'Tokenized text example: {tokenized_text_ex}')

    answer_tokens = glove_tokenizer.texts_to_sequences(texts_list)
    answer_tokens = pad_sequences(answer_tokens, maxlen=max_token_length, padding='post', truncating='post')

    answer_lemmatized_tokens = glove_tokenizer.texts_to_sequences(lemmatized_texts_list)
    answer_lemmatized_text = glove_tokenizer.sequences_to_texts(answer_lemmatized_tokens)
    # answer_lemmatized_tokens = [[i for i in j if i < len(glove_tokenizer.word_index)] for j in answer_lemmatized_tokens]
    answer_lemmatized_tokens = pad_sequences(answer_lemmatized_tokens, maxlen=max_token_length, padding='post', truncating='post')
    # create mask
    # change to lemmatized mask
    answer_mask = torch.zeros(answer_lemmatized_tokens.shape, dtype=torch.long)
    # print(torch.Tensor(answer_tokens))
    # FIXME: this is a hack to get the mask to work, I'm going to remove the part that makes it a tensor: answer_mask = torch.Tensor(answer_tokens)
    answer_mask.masked_fill_(torch.Tensor(answer_lemmatized_tokens) != 0, 1)

    keyword_tokens = glove_tokenizer.texts_to_sequences(keywords)
    keyword_tokens = pad_sequences(keyword_tokens, maxlen=max_keyword_length, padding='post', truncating='post')
    keyword_tokens = torch.reshape(torch.from_numpy(keyword_tokens), (len(keywords), max_keyword_length))
    keyword_mask = torch.zeros(keyword_tokens.shape, dtype=torch.long)
    keyword_mask.masked_fill_(keyword_tokens != 0, 1)

    return TorchTabularTextDataset(hf_model_text_input, categorical_feats,
                                numerical_feats, answer_tokens, answer_mask, keyword_tokens, keyword_mask, [0, 1, 2, 3, 4], data_df, label_list=None, class_weights=None,
                                texts=texts_list,
                                lemmatized_answer_tokens=answer_lemmatized_tokens, lemmatized_answer_texts=lemmatized_texts_list, answer_lemmatized_text=answer_lemmatized_text)


def load_data(data_df,
              text_cols,
              tokenizer,
              label_col,
              label_list=None,
              categorical_cols=None,
              numerical_cols=None,
              sep_text_token_str=' ',
              categorical_encode_type='ohe',
              numerical_transformer=None,
              empty_text_values=None,
              replace_empty_text=None,
              max_token_length=None,
              debug=False, glove_tokenizer=None,
              keywords=None,
              max_keyword_length=20
              ):
    """Function to load a single dataset given a pandas DataFrame

    Given a DataFrame, this function loads the data to a :obj:`torch_dataset.TorchTextDataset`
    object which can be used in a :obj:`torch.utils.data.DataLoader`.

    Args:
        data_df (:obj:`pd.DataFrame`): The DataFrame to convert to a TorchTextDataset
        text_cols (:obj:`list` of :obj:`str`): the column names in the dataset that contain text
            from which we want to load
        tokenizer (:obj:`transformers.tokenization_utils.PreTrainedTokenizer`):
            HuggingFace tokenizer used to tokenize the input texts as specifed by text_cols
        label_col (str): The column name of the label, for classification the column should have
            int values from 0 to n_classes-1 as the label for each class.
            For regression the column can have any numerical value
        label_list (:obj:`list` of :obj:`str`, optional): Used for classification;
            the names of the classes indexed by the values in label_col.
        categorical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that
            contain categorical features. The features can be already prepared numerically, or
            could be preprocessed by the method specified by categorical_encode_type
        numerical_cols (:obj:`list` of :obj:`str`, optional): The column names in the dataset that contain numerical features.
            These columns should contain only numeric values.
        sep_text_token_str (str, optional): The string token that is used to separate between the
            different text columns for a given data example. For Bert for example,
            this could be the [SEP] token.
        categorical_encode_type (str, optional): Given categorical_cols, this specifies
            what method we want to preprocess our categorical features.
            choices: [ 'ohe', 'binary', None]
            see encode_features.CategoricalFeatures for more details
        numerical_transformer (:obj:`sklearn.base.TransformerMixin`): The sklearn numeric
            transformer instance to transform our numerical features
        empty_text_values (:obj:`list` of :obj:`str`, optional): Specifies what texts should be considered as
            missing which would be replaced by replace_empty_text
        replace_empty_text (str, optional): The value of the string that will replace the texts
            that match with those in empty_text_values. If this argument is None then
            the text that match with empty_text_values will be skipped
        max_token_length (int, optional): The token length to pad or truncate to on the
            input text
        debug (bool, optional): Whether or not to load a smaller debug version of the dataset
        glove_tokenizer (:obj, optional): The original tokenizer is provided by huggingface, this is any tokenizer
        object provided by torchtext to prepare the tokens for the glove embedding layer

    Returns:
        :obj:`tabular_torch_dataset.TorchTextDataset`: The converted dataset
    """
    if debug:
        data_df = data_df[:500]
    if empty_text_values is None:
        empty_text_values = ['nan', 'None']

    text_cols_func = convert_to_func(text_cols)
    categorical_cols_func = convert_to_func(categorical_cols)
    numerical_cols_func = convert_to_func(numerical_cols)

    categorical_feats, numerical_feats = load_cat_and_num_feats(data_df,
                                                                categorical_cols_func,
                                                                numerical_cols_func,
                                                                categorical_encode_type)
    numerical_feats = normalize_numerical_feats(numerical_feats, numerical_transformer)
    agg_func = partial(agg_text_columns_func, empty_text_values, replace_empty_text)
    texts_cols = get_matching_cols(data_df, text_cols_func)
    logger.info(f'Text columns: {texts_cols}')

    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    lemmatized_texts_list = data_df[['lemmatized']].agg(agg_func, axis=1).tolist()

    for i, text in enumerate(texts_list):
        texts_list[i] = f' {sep_text_token_str} '.join(text)

    for i, text in enumerate(lemmatized_texts_list):
        lemmatized_texts_list[i] = f' {sep_text_token_str} '.join(text)

    logger.info(f'Raw text example: {texts_list[0]}')
    logger.info(f'Lemmatized text example: {lemmatized_texts_list[0]}')

    hf_model_text_input = tokenizer(texts_list, padding="max_length", truncation=True,
                                    max_length=max_token_length)
    tokenized_text_ex = ' '.join(tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
    logger.debug(f'Tokenized text example: {tokenized_text_ex}')
    labels = data_df[label_col].values

    answer_tokens = glove_tokenizer.texts_to_sequences(texts_list)
    answer_tokens = pad_sequences(answer_tokens, maxlen=max_token_length, padding='post', truncating='post')

    answer_lemmatized_tokens = glove_tokenizer.texts_to_sequences(lemmatized_texts_list)
    answer_lemmatized_tokens = pad_sequences(answer_lemmatized_tokens, maxlen=max_token_length, padding='post', truncating='post')

    # create mask
    # change to lemmatized mask
    answer_mask = torch.zeros(answer_lemmatized_tokens.shape, dtype=torch.long)
    # print(torch.Tensor(answer_tokens))
    # FIXME: this is a hack to get the mask to work, I'm going to remove the part that makes it a tensor: answer_mask = torch.Tensor(answer_tokens)
    answer_mask.masked_fill_(torch.Tensor(answer_lemmatized_tokens) != 0, 1)

    keyword_tokens = glove_tokenizer.texts_to_sequences(keywords)
    keyword_tokens = pad_sequences(keyword_tokens, maxlen=max_keyword_length, padding='post', truncating='post')
    keyword_tokens = torch.reshape(torch.from_numpy(keyword_tokens), (len(keywords), max_keyword_length))
    keyword_mask = torch.zeros(keyword_tokens.shape, dtype=torch.long)
    keyword_mask.masked_fill_(keyword_tokens != 0, 1)

    return TorchTabularTextDataset(hf_model_text_input, categorical_feats,
                                   numerical_feats, answer_tokens, answer_mask, keyword_tokens, keyword_mask, labels, data_df, label_list, texts=texts_list, lemmatized_answer_tokens=answer_lemmatized_tokens, lemmatized_answer_texts=lemmatized_texts_list)

