import pickle
import re

import nltk
import pandas as pd
from nltk import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.corpus.reader.wordnet as wordnet


def get(filename, genres):
    data = pd.read_csv(filename, encoding='utf8')
    data = data[['Plot', 'Genre']]
    data = data[data['Genre'].isin(genres)]
    df = data['Plot']
    y = data['Genre'].apply(lambda x: 0 if x == 'comedy' else 1)
    return df, y


stemmer = SnowballStemmer(language='english')
stop_words = set(stopwords.words('english'))


lemmatizer =  WordNetLemmatizer()

ADJECTIVE = 'JJ'
ADVERB = 'RB'
VERB = 'VB'
NOUN = 'NN'
NOUN_PROPER = 'NNP'
TAG_INDEX = 1
WORD_INDEX = 0

lemma_tag = {
    ADJECTIVE: wordnet.ADJ,
    ADVERB: wordnet.ADV,
    VERB: wordnet.VERB,
    NOUN: wordnet.NOUN
}

tag_list = [NOUN, VERB, ADVERB, ADJECTIVE]

def pos_tag_filter(words_tag):
    filtered = []
    for word_tag in words_tag:
        if word_tag[TAG_INDEX] == NOUN_PROPER:
            continue
        for tag in tag_list:
            if tag in word_tag[TAG_INDEX] and word_tag[WORD_INDEX] not in stop_words:
                filtered.append(
                    (
                        word_tag[WORD_INDEX].lower(),
                        lemma_tag[tag]
                    )
                )
    return filtered

def analyze(sentence: str):
    tokenizer = TweetTokenizer().tokenize
    tokens = tokenizer(sentence)
    words_tag = nltk.pos_tag(tokens)
    filtered_words_tag = pos_tag_filter(words_tag)
    lemmas = [lemmatizer.lemmatize(word[WORD_INDEX], pos=word[TAG_INDEX])
              for word in filtered_words_tag]
    return lemmas

def analyze_simple(sentence: str):
    tokenizer = TweetTokenizer().tokenize
    tokens = tokenizer(sentence)
    return tokens

def pre_process(doc):
    return ' '.join(analyze(doc))

def unpickle(filename):
    with open(filename) as pkl:
        df, y = pickle.load(pkl)
        return df, y

def read(filename):
    data = pd.read_csv(filename, encoding='utf8')
    df = data['Plot']
    y = data['Genre']
    return df, y