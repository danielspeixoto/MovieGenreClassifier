from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import data
from DenseTransformer import DenseTransformer


class Run:

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def pipeline(self):
        text_clf = Pipeline([
            ('vect', CountVectorizer(
                analyzer=data.analyze_simple
            )),
            ('tfidf', TfidfTransformer()),
            ('dense', DenseTransformer()),
            ('clf', self.algorithm)
        ])
        return text_clf

    @staticmethod
    def bayes():
        return MultinomialNB()

    @staticmethod
    def svm(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42):
        return SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            n_iter=n_iter,
            random_state=random_state
        )

    def params(self):
        print("----Classifier: " + type(self.algorithm).__name__)
