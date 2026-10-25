import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf

KERAS_MODEL_PATH = Path("artifacts/fashion_mnist.keras")
ONNX_MODEL_PATH = Path("artifacts/fashion_mnist.onnx")
RESULTS_PATH = Path("results/onnx_verification.json")

# Load test and normalize test data
(_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

assert test_images.shape == (10_000, 28, 28)
assert test_images.dtype == "uint8"
assert test_labels.shape == (10_000,)
assert test_labels.dtype == "uint8"

# Export test data as raw binary files for Go model
test_images.tofile("artifacts/fashion_mnist_test_images.bin")
test_labels.tofile("artifacts/fashion_mnist_test_labels.bin")

test_images = test_images.astype("float32") / 255.0

# Verify the test data
assert test_images.shape == (10000, 28, 28)
assert test_images.dtype == "float32"
assert test_images.min() >= 0.0
assert test_images.max() <= 1.0

# Load the trained Keras model
model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    compile=False,
)

assert model.input_shape == (None, 28, 28)
assert model.output_shape == (None, 10)

# Export to ONNX
model.export(
    ONNX_MODEL_PATH,
    format="onnx",
    input_signature=[
        tf.TensorSpec(
            shape=(None, 28, 28),
            dtype=tf.float32,
            name="image",
        )
    ],
)

onnx_model = onnx.load(ONNX_MODEL_PATH)
onnx.checker.check_model(onnx_model)

onnx_opset_version = onnx_model.opset_import[0].version

print("ONNX model passed structural validation")
print("ONNX opset:", onnx_model.opset_import[0].version)

# Create an ONNX Runtime session
session = ort.InferenceSession(
    str(ONNX_MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("Input name:", input_info.name)
print("Input shape:", input_info.shape)
print("Input type:", input_info.type)

print("Output name:", output_info.name)
print("Output shape:", output_info.shape)
print("Output type:", output_info.type)

# Obtain Keras and ONNX outputs
keras_logits = model.predict(
    test_images,
    batch_size=32,
    verbose=0,
)

onnx_logits = session.run(
    [output_info.name],
    {input_info.name: test_images},
)[0]

assert keras_logits.shape == (10000, 10)
assert onnx_logits.shape == keras_logits.shape

# Convert logits into predicted classes
keras_classes = np.argmax(keras_logits, axis=1)
onnx_classes = np.argmax(onnx_logits, axis=1)

class_agreement = float(np.mean(keras_classes == onnx_classes))
keras_accuracy = float(np.mean(keras_classes == test_labels))
onnx_accuracy = float(np.mean(onnx_classes == test_labels))

maximum_absolute_difference = float(np.max(np.abs(keras_logits - onnx_logits)))

print("Keras accuracy:", keras_accuracy)
print("ONNX accuracy:", onnx_accuracy)
print("Class agreement:", class_agreement)
print("Maximum difference:", maximum_absolute_difference)

verification_results = {
    "onnx_opset_version": onnx_opset_version,
    "input_name": input_info.name,
    "input_shape": input_info.shape,
    "input_type": input_info.type,
    "output_name": output_info.name,
    "output_shape": output_info.shape,
    "output_type": output_info.type,
    "keras_accuracy": keras_accuracy,
    "onnx_accuracy": onnx_accuracy,
    "class_agreement": class_agreement,
    "maximum_absolute_difference": maximum_absolute_difference,
}

RESULTS_PATH.parent.mkdir(exist_ok=True)
RESULTS_PATH.write_text(
    json.dumps(verification_results, indent=2) + "\n",
    encoding="utf-8",
)
