from evaluation.metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score

def test_accuracy():
    actual=[1,0,1,1]
    predicted=[1,0,0,1]

    result=calculate_accuracy(actual,predicted)

    assert result==0.75

def test_precision():
    actual=[1,0,1,0]
    predicted=[1,1,1,1]

    result=calculate_precision(actual,predicted)

    assert result== 0.5

def test_recall():
    actual=[1,1,1,0]
    predicted=[1,0,1,0]

    result=calculate_recall(actual,predicted)

    assert round(result, 4)== round(2/3, 4)

def test_f1_score():
    actual=[1,0,1,0]
    predicted=[1,1,1,0]

    result=calculate_f1_score(actual,predicted)

    assert round(result, 4)== round(0.8, 4)