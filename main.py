from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC

import data
import validation
from Run import Run

print("Loading data")
df, y = data.read("pre_processed.csv")
print("Data Loaded")
configs = [
    {
        "parameters": {
            'vect__ngram_range': [(1, 1)]
        },
        "algorithm": MultinomialNB()
    },
    {
        "parameters": {
            # 'vect__ngram_range': [(1, 1), (1, 2)]
            # 'clf__gamma': (1e-3, 1e-4),
            'clf__C': [10],
            'clf__kernel': ['linear'],
            # 'max_iter'
        },
        "algorithm": SVC(
            max_iter=-1,
            kernel='linear',
            verbose=True,
            gamma="auto"
        )
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

        # text_clf = GridSearchCV(run.pipeline(), config['parameters'], n_jobs=1, cv=3, verbose=True)
        text_clf = run.pipeline()
        # text_clf.fit(df, y)
        # print("Params")
        # print(text_clf.best_params_)
        # print("score")
        # print(text_clf.best_score_)

        accuracy = validator(df, y, text_clf)
        print("--------Accuracy: %.2f" % accuracy)
        print("##########################################")
    print("**************************************")
