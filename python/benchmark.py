import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Identify and load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = fashion_mnist.load_data()


def print_image_statistics(name, images):
    print(f"{name} data type:", images.dtype)
    print(f"{name} shape:", images.shape)
    print(f"{name} minimum pixel value:", images.min())
    print(f"{name} maximum pixel value:", images.max())


print_image_statistics("Test images before normalization", test_images)

test_images = test_images.astype("float32") / 255.0


def assert_normalized_images(name, images, labels):
    assert images.dtype == "float32", f"{name} must use float32"
    assert images.min() >= 0.0, f"{name} contains a value below 0.0"
    assert images.max() <= 1.0, f"{name} contains a value above 1.0"
    assert images.ndim == 3, f"{name} must have three dimensions"
    assert images.shape[1:] == (28, 28), f"{name} images must be 28 by 28"
    assert images.shape[0] == labels.shape[0], f"{name} must have one label per image"


assert_normalized_images("Test images", test_images, test_labels)
print_image_statistics("Test images after normalization", test_images)

model = tf.keras.models.load_model("artifacts/fashion_mnist.keras")

assert model.input_shape == (None, 28, 28)
assert model.output_shape == (None, 10)

BATCH_SIZE = 32
REPETITIONS = 5

# Warm up the model with one initial run
model.predict(
    test_images,
    batch_size=BATCH_SIZE,
    verbose=0,
)

durations = []
logits = None

# Time five inference runs
for repetition in range(REPETITIONS):
    start_time = time.perf_counter()

    logits = model.predict(
        test_images,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    elapsed_time = time.perf_counter() - start_time
    durations.append(elapsed_time)

    print(f"Run {repetition + 1}: {elapsed_time:.4f} seconds")

# Convert logits into classes
predicted_classes = np.argmax(logits, axis=1)

assert logits.shape == (10000, 10)
assert predicted_classes.shape == test_labels.shape

# Calculate accuracy
accuracy = np.mean(predicted_classes == test_labels)

# Convert to a standard Python number for JSON
accuracy = float(accuracy)

# Calculate average time and throughput
average_duration = sum(durations) / len(durations)
throughput = len(test_images) / average_duration

print("Accuracy:", accuracy)
print("Average duration:", average_duration)
print("Throughput:", throughput, "images per second")

# Create results directory
results_directory = Path("results")
results_directory.mkdir(exist_ok=True)

# Save the predictions
np.savetxt(
    results_directory / "python_predictions.csv",
    predicted_classes,
    fmt="%d",
    delimiter=",",
)

# Create a dictionary to save the benchmark results
benchmark_results = {
    "test_images": len(test_images),
    "batch_size": BATCH_SIZE,
    "warmup_runs": 1,
    "repetitions": REPETITIONS,
    "durations_seconds": durations,
    "average_duration_seconds": average_duration,
    "throughput_images_per_second": throughput,
    "accuracy": accuracy,
}

with open(
    results_directory / "python_benchmark.json",
    "w",
    encoding="utf-8",
) as results_file:
    json.dump(benchmark_results, results_file, indent=2)
