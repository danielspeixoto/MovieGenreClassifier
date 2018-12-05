from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import data
import validation
from Run import Run

df, y = data.read("movies-pre_processed.csv")
configs = [
    {
        "parameters": {
            'tfidf__use_idf': [True, False],
            'vect__ngram_range': [(1, 1)],
           #'clf__alpha': [0.7],
        },
        "algorithm": MultinomialNB()
    },
    # {
    #     "parameters": {
    #         "clf__n_neighbors": [5],
    #         # "clf__weights": ['distance'],
    #         'clf__metric': ['cosine']
    #     },
    #     "algorithm": KNeighborsClassifier(n_jobs=-1)
    #}
    {
        "parameters": {
            'tfidf__use_idf': [True, False],
            'vect__ngram_range': [(1, 1)],
           'clf__criterion': ['entropy'],
        },
        "algorithm": DecisionTreeClassifier()
    },
]
#
validations = [
    # K Fold
    lambda df, y, clf: validation.k_fold(10, df, y, clf),
]

for validator in validations:
    for config in configs:
        run = Run(algorithm=config["algorithm"])
        run.params()

        text_clf = run.pipeline()
        text_clf.fit(df, y)

        results = validator(df, y, text_clf)
        print("--------Accuracy: %.2f" % results[0])
        print("--------Precision: %.2f" % results[1])
        print("--------Recall: %.2f" % results[2])
        print("--------F-Score: %.2f" % results[3])
        print("##########################################")
        print("**************************************")
