from evaluation.metrics import(
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score
)

def run_benchmark(actual, predicted):
    accuracy = calculate_accuracy(actual, predicted)
    precision = calculate_precision(actual, predicted)
    recall = calculate_recall(actual, predicted)
    f1 = calculate_f1_score(actual, predicted)

    return{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":
    actual = [1, 0, 1, 1]
    predicted = [1, 0, 0, 1]

    benchmark_results = run_benchmark(actual, predicted)
    print(benchmark_results)