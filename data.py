import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def get(filename, genres):
    data = pd.read_csv(filename, encoding='utf8')
    data = data[['Plot', 'Genre']]
    data = data[data['Genre'].isin(genres)]
    df = data['Plot']
    # TODO
    y = data['Genre'].apply(lambda x: 0 if x == 'comedy' else 1)
    return df, y


stemmer = SnowballStemmer(language='english')
stop_words = set(stopwords.words('english'))

def analyze(doc):
    tokens = word_tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    return stemmed
