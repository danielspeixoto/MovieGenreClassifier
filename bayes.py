import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import data
from DenseTransformer import DenseTransformer

class Run:

    def __init__(self, ngram_max, algorithm):
        self.ngram_max = ngram_max
        self.algorithm = algorithm

    def pipeline(self):
        text_clf = Pipeline([
            ('vect', CountVectorizer(
                analyzer=data.analyze,
                ngram_range=(1, self.ngram_max)
            )),
            ('tfidf', TfidfTransformer(
                sublinear_tf=True,
                norm='l2'
            )),
            ('dense', DenseTransformer()),
            ('clf', self.algorithm)
        ])
        return text_clf

    @staticmethod
    def bayes():
        return MultinomialNB()

    def params(self):
        print("Params")
        print("----Classifier: " + type(self.algorithm).__name__)
        print("----NGrams: " + str(self.ngram_max))
