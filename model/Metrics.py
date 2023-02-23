from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def macro_f1(y_true, y_pred):
    # Macro -> Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    return f1_score(y_true, y_pred, average='macro')

def confusion_matrix_sk(y_true, y_pred, labels=None):
    return confusion_matrix(y_true, y_pred)