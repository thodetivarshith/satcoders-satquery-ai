import numpy as np
import time

from optical_sar_fusion import (
    fuse_optical_sar,
    convert_to_hwc
)


def test_performance():

    print("\n===== WEEK 3: PERFORMANCE TEST =====")

    optical = np.random.rand(4, 256, 256)
    sar = np.random.rand(2, 256, 256)

    start_time = time.perf_counter()

    

    fused = fuse_optical_sar(optical, sar)

    final = convert_to_hwc(fused)

    end_time = time.perf_counter()

    processing_time = end_time - start_time
    processing_time_ms = processing_time * 1000

    print("Optical Shape:", optical.shape)
    print("SAR Shape:", sar.shape)
    print("Fused Shape:", fused.shape)
    print("Final HWC Shape:", final.shape)

    print(f"\nProcessing Time: {processing_time_ms:.4f} ms")

    if processing_time_ms < 500:
        print("Performance Test: PASSED (< 500 ms)")
    else:
        print("Performance Test: FAILED (> 500 ms)")


def test_nan_values():

    print("\n===== NaN VALUE TEST =====")

    optical = np.random.rand(4, 256, 256)
    sar = np.random.rand(2, 256, 256)

    optical[0, 0, 0] = np.nan

    try:

        if np.isnan(optical).any() or np.isnan(sar).any():
            raise ValueError("NaN values detected in input data")

        fused = fuse_optical_sar(optical, sar)

    except ValueError as error:
        print("Correctly detected NaN values!")
        print("Error:", error)


def test_infinite_values():

    print("\n===== INFINITE VALUE TEST =====")

    optical = np.random.rand(4, 256, 256)
    sar = np.random.rand(2, 256, 256)

    sar[0, 0, 0] = np.inf

    try:

        if np.isinf(optical).any() or np.isinf(sar).any():
            raise ValueError("Infinite values detected in input data")

        fused = fuse_optical_sar(optical, sar)

    except ValueError as error:
        print("Correctly detected infinite values!")
        print("Error:", error)


def test_empty_input():

    print("\n===== EMPTY INPUT TEST =====")

    optical = np.array([])
    sar = np.array([])

    try:

        fuse_optical_sar(optical, sar)

    except ValueError as error:
        print("Correctly detected invalid empty input!")
        print("Error:", error)


def test_multiple_performance():

    print("\n===== MULTIPLE IMAGE PERFORMANCE TEST =====")

    total_time = 0
    number_of_samples = 10

    for i in range(number_of_samples):

        optical = np.random.rand(4, 256, 256)
        sar = np.random.rand(2, 256, 256)

        start_time = time.perf_counter()

        fused = fuse_optical_sar(optical, sar)
        final = convert_to_hwc(fused)

        end_time = time.perf_counter()

        processing_time_ms = (
            end_time - start_time
        ) * 1000

        total_time += processing_time_ms

        print(
            f"Sample {i + 1}: "
            f"{processing_time_ms:.4f} ms"
        )

    average_time = total_time / number_of_samples

    print(
        f"\nAverage Processing Time: "
        f"{average_time:.4f} ms"
    )

    if average_time < 500:
        print("Average Performance: PASSED (< 500 ms)")
    else:
        print("Average Performance: FAILED (> 500 ms)")


if __name__ == "__main__":

    print("===== PRODUCTION READINESS TESTING =====")

    test_performance()
    test_nan_values()
    test_infinite_values()
    test_empty_input()
    test_multiple_performance()