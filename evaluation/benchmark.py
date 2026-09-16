import json
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

def print_benchmark_results(results):
    print("\n===== SatQuery AI Benchmark Results =====")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")

def save_benchmark_results(results):
    with open("evaluation/results/benchmark_results.json", "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    actual = [1, 0, 1, 1]
    predicted = [1, 0, 0, 1]

    benchmark_results = run_benchmark(actual, predicted)

    print_benchmark_results(benchmark_results)

    save_benchmark_results(benchmark_results)