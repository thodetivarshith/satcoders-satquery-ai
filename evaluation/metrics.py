from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def calculate_accuracy(actual,predicted):
    return accuracy_score(actual,predicted)


def calculate_precision(actual,predicted):
    return precision_score(actual,predicted,zero_division=0)

def calculate_recall(actual,predicted):
    return recall_score(actual,predicted,zero_division=0)

def calculate_f1_score(actual,predicted):
    return f1_score(actual,predicted,zero_division=0)