import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics

def k_fold(k, df, y, clf):
    print("----Validation: " + str(k) + "-Fold")
    fold = StratifiedKFold(k)
    final_accuracy = 0.0
    final_precision = 0.0
    final_recall = 0.0
    final_fscore = 0.0
    for train, test in fold.split(df, y):
        X_train, X_test = df.iloc[train], df.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = np.mean(predicted == y_test)
        print("--------Current Accuracy: %.2f" % accuracy)
        final_accuracy = final_accuracy + accuracy
        precision = metrics.precision_score(y_test, predicted)
        print("--------Current Precision: %.2f" % precision)
        final_precision = final_precision + precision
        recall = metrics.recall_score(y_test, predicted)
        print("--------Current Recall: %.2f" % recall)
        final_recall = final_recall + recall
        fscore = metrics.f1_score(y_test, predicted)
        print("--------Current F-Score: %.2f" % fscore)
        final_fscore = final_fscore + fscore
        print("##########################################")
        

    return [final_accuracy/k, 
            final_precision/k,
            final_recall/k,
            final_fscore,]

def loss(y, y_pred):
    result = np.sum(y == y_pred)/float(len(y_pred))
    print("--------Current Accuracy: " + str(result))
    return result
