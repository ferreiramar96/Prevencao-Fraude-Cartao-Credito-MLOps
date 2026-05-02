from sklearn.metrics import (
    f1_score, accuracy_score, recall_score, 
    precision_score, roc_auc_score)
import sklearn


def evaluate_model(model:sklearn, x_test, y_test):
    result = model.predict(x_test)

    metrics = {
        "f1_score": f1_score(y_test, result),
        "accuracy": accuracy_score(y_test, result),
        "recall": recall_score(y_test, result),
        "precision": precision_score(y_test, result),
        "auc_roc": roc_auc_score(y_test, result)
    }

    return metrics