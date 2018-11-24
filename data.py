import pickle
import re

import nltk
import pandas as pd
from nltk import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def get(filename, genres):
    data = pd.read_csv(filename, encoding='utf8')
    data = data[['Plot', 'Genre']]
    data = data[data['Genre'].isin(genres)]
    df = data['Plot']
    y = data['Genre'].apply(lambda x: 0 if x == 'comedy' else 1)
    return df, y


stemmer = SnowballStemmer(language='english')
stop_words = set(stopwords.words('english'))

ADJECTIVE = 'JJ'
ADVERB = 'RB'
VERB = 'VB'
TAG_INDEX = 1
WORD_INDEX = 0
def pos_tag_filter(words_tag):
    filtered = [word_tag for word_tag in words_tag
                if ADJECTIVE in word_tag[TAG_INDEX] or
                ADVERB in word_tag[TAG_INDEX]
                ]
    return filtered

def analyze(sentence: str):
    tokenizer = TweetTokenizer().tokenize
    tokens = tokenizer(sentence)
    tokens = [token for token in tokens if re.match(r"[A-Za-z]", token)]
    tokens = [token.lower() for token in tokens]
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    return stemmed

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