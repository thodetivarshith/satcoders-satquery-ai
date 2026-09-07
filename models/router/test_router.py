# Router tests - Abhinay
"""
Automated tests for SatQuery AI Query Classifier.
"""

from query_classifier import classify_query


TEST_CASES = [
    # Grounding
    ("Where are the buildings?", "grounding"),
    ("Locate all buildings in the image", "grounding"),
    ("Find the roads", "grounding"),
    ("Show me the water bodies", "grounding"),
    ("Where is the airport?", "grounding"),
    ("Identify the road network", "grounding"),
    ("Point out the river", "grounding"),

    # VQA
    ("What is visible in the image?", "vqa"),
    ("Describe this satellite image", "vqa"),
    ("How many buildings are there?", "vqa"),
    ("What objects are visible?", "vqa"),
    ("Which areas are urban?", "vqa"),
    ("Is there a lake in the image?", "vqa"),

    # Change Detection
    ("What changed between the two images?", "change_detection"),
    ("Compare the two images", "change_detection"),
    ("Did the forest area change?", "change_detection"),
    ("What is different between these images?", "change_detection"),
    ("Has the urban area increased?", "change_detection"),
    ("Compare the before and after images", "change_detection"),

    # Fusion
    ("Analyze Sentinel-1 SAR data", "fusion"),
    ("Use optical and SAR imagery", "fusion"),
    ("Analyze fused satellite data", "fusion"),
    ("Use multispectral and SAR data", "fusion"),
]


def run_tests():
    """Run all classifier test cases."""

    passed = 0
    failed = 0

    print("\nRunning Query Classifier Tests...\n")

    for query, expected_task in TEST_CASES:

        result = classify_query(query)
        predicted_task = result["task"]

        if predicted_task == expected_task:
            print(f"PASS: {query}")
            print(f"      → {predicted_task} "
                  f"(confidence={result['confidence']})")
            passed += 1

        else:
            print(f"FAIL: {query}")
            print(f"      Expected: {expected_task}")
            print(f"      Got:      {predicted_task}")
            failed += 1

    total = passed + failed
    accuracy = (passed / total) * 100

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Accuracy    : {accuracy:.2f}%")
    print("=" * 50)

    if accuracy >= 80:
        print("SUCCESS: Router accuracy requirement achieved!")
    else:
        print("WARNING: Router accuracy is below 80%.")


if __name__ == "__main__":
    run_tests()
    