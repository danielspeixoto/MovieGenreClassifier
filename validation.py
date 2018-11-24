import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def k_fold(k, df, y, clf):
    print("----Validation: " + str(k) + "-Fold")
    fold = StratifiedKFold(k)
    final = 0.0
    for train, test in fold.split(df, y):
        X_train, X_test = df.iloc[train], df.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = np.mean(predicted == y_test)
        print("--------Current Accuracy: %.2f" % accuracy)
        final = final + accuracy

    return final/k
