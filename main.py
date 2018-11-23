import data
import validation
from bayes import Run

df, y = data.get("movies.csv", ['comedy', 'drama'])

configs = [
    {
        "ngram_max": 1,
        "algorithm": Run.bayes()
    },
    {
        "ngram_max": 2,
        "algorithm": Run.bayes()
    }
]

validations = [
    # 10 Fold
    lambda df, y, clf: validation.k_fold(10, df, y, clf),
    # 5 Fold
    lambda df, y, clf: validation.k_fold(5, df, y, clf),
    # 3 Fold
    lambda df, y, clf: validation.k_fold(3, df, y, clf),
]

# TODO Parallel
for validator in validations:
    for config in configs:
        run = Run(ngram_max=config["ngram_max"],
                  algorithm=config["algorithm"])
        run.params()

        text_clf = run.pipeline()

        accuracy = validator(df, y, text_clf)
        print("--------Accuracy: %.2f" % accuracy)
        print("##########################################")
    print("**************************************")
