from sklearn.metrics import roc_curve, roc_auc_score


# for logistic regression: auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
def roc_auc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, thresholds, auc


# TODO F1, Precision, Recall after deciding on the probability threshold.
