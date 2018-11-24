from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

import data
import validation
from Run import Run

df, y = data.read("movies-pre_processed.csv")
configs = [
    # {
    #     "parameters": {
    #         'tfidf__use_idf': [False],
    #         # 'vect__ngram_range': [(1, 2)],
    #         'clf__alpha': [0.85],
    #     },
    #     "algorithm": MultinomialNB()
    # },
    # {
    #     "parameters": {
    #         # 'vect__ngram_range': [(1, 1), (1, 2)]
    #         # 'clf__gamma': 'scale',
    #         # 'clf__C': [1, 100],
    #         'clf__kernel': ['linear'],
    #         # 'max_iter'
    #     },
    #     "algorithm": SVC(
    #         tol=1e-1,
    #         max_iter=-1,
    #         cache_size=10000
    #     )
    # },
    {
        "parameters": {
            "clf__n_neighbors": [3, 5, 7]
        },
        "algorithm": KNeighborsClassifier()
    }
]
#
validations = [
    # K Fold
    lambda df, y, clf: validation.k_fold(3, df, y, clf),
]

for validator in validations:
    for config in configs:
        run = Run(algorithm=config["algorithm"])
        run.params()

        # text_clf = run.pipeline()
        text_clf = GridSearchCV(run.pipeline(), config['parameters'],
                                n_jobs=-1, cv=3, verbose=True,
                                scoring=make_scorer(
                                    validation.loss,
                                    greater_is_better=True))
        text_clf.fit(df, y)
        print("Params")
        print(text_clf.best_params_)
        print("score")
        print(text_clf.best_score_)

        # accuracy = validator(df, y, text_clf)
        # print("--------Accuracy: %.2f" % accuracy)
        print("##########################################")
    print("**************************************")
