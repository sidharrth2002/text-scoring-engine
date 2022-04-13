import spacy
import re
import string
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy
from spellchecker import SpellChecker
import textstat

nlp = spacy.load('en_core_web_sm')

spell = SpellChecker()
english = English()
tokenizer = Tokenizer(english.vocab)

def num_words(text):
    # find number of words in text
    doc = nlp(text)
    return len(doc)

def num_WORDS(text):
    # find number of words in text
    doc = nlp(text)
    return len(doc)

def num_sentences(text):
    # find number of sentences in text
    doc = nlp(text)
    return len(list(doc.sents))

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def num_lemmas(text):
    # find number of lemmas in text
    return len(lemmatize(text))

def count_commas(text):
  count = 0  
  for i in range (0, len(text)):   
    if text[i] == ',':  
        count = count + 1
  return count

def count_exclamation_marks(text):
  count = 0 
  for i in range (0, len(text)):   
    if text[i] == '!':  
        count = count + 1
  return count

def count_question_marks(text):
    count = 0
    for i in range (0, len(text)):
        if text[i] == '?':  
            count = count + 1
    return count

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def average_word_length(text):
    sentences = [sent.text for sent in nlp(text).sents]
    length_words = 0
    total_words = 0
    for sentence in sentences:
        words = tokenizer(sentence)
        for word in words:
            length_words += len(word)
            total_words += 1
    return length_words / total_words

def average_sentence_length(text):
    sentences = [sent.text for sent in nlp(text).sents]
    length_sentences = 0
    total_sentences = 0
    for sentence in sentences:
        length_sentences += len(tokenizer(sentence))
        total_sentences += 1
    return length_sentences / total_sentences

def number_of_nouns(text):
    # find number of nouns in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('NOUN') + pos.count('PROPN')

def number_of_verbs(text):
    # find number of verbs in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('VERB')

def number_of_adverbs(text):
    # find number of adverbs in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('ADV')

def number_of_adjectives(text):
    # find number of adjectives in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('ADJ')

def number_of_conjunctions(text):
    # find number of conjunctions in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('CCONJ')

def number_of_spelling_errors(text):
    misspelled = spell.unknown([token.text for token in tokenizer(text)])
    return len(misspelled)

def num_stopwords(text):
    # find number of stopwords in text
    doc = nlp(text)
    stop_words = [token.text for token in doc if token.is_stop]
    return len(stop_words)

def num_ORG(text):
    # find number of organizations in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('ORG')

def num_LOC(text):
    # find number of locations in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('LOC')

def num_PER(text):
    # find number of people in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('PER')

def num_MISC(text):
    # find number of mixed in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('MISC')

def num_GPE(text):
    # find number of mixed in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('GPE')

def num_FAC(text):
    # find number of mixed in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('FAC')

def num_PRODUCT(text):
    # find number of mixed in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('PRODUCT')

def num_EVENT(text):
    # find number of mixed in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('EVENT')

def num_LAW(text):
    # find number of mixed in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('LAW')

def automated_readability_index(text):
    # find automated readability index in text
    return textstat.automated_readability_index(text)

def coleman_liau_index(text):
    # find coleman liau index in text
    return textstat.coleman_liau_index(text)

def dale_chall_index(text):
    # find dale chall readability score in text
    return textstat.dale_chall_readability_score(text)

def difficult_word_count(text):
    # find difficult word count in text
    return textstat.difficult_words(text)

def flesch_kincaid_grade(text):
    # find flesch kincaid grade in text
    return textstat.flesch_kincaid_grade(text)

def gunning_fog(text):
    # find gunning fog index in text
    return textstat.gunning_fog(text)

def linsear_write_formula(text):
    # find linsear write formula in text
    return textstat.linsear_write_formula(text)

def smog_index(text):
    # find smog index in text
    return textstat.smog_index(text)

def syllable_count(text):
    # find syllable count in text
    return textstat.syllable_count(text)