# Fashion MNIST Python and GoMLX Comparison

## Objective
This project compares inference performance between a TensorFlow/Keras deep learning model in Python and a GoMLX model using the Fashion-MNIST dataset. The trained Keras model was exported to ONNX and loaded into GoMLX to benchmark both models and compare their predictions.

The model was trained on 54,000 images while 6,000 were reserved for validation. The final 10,000 test images were used to evaluate how well the model classified images into the ten Fashion-MNIST categories.

The model architecture was adapted from TensorFlow's Fashion-MNIST classification tutorial. The model was trained and benchmarked locally for this project.

## Dataset
Fashion-MNIST contains 70,000 grayscale images labeled under 10 numeric labels which correspond to ten clothing categories:

* Images use grayscale pixel value from `0` to `255`
* The original image data type is `uint8`
* Each image has 28 by 28 pixels or contains 784 pixels total

The training set has 60,000 images and the testing set has 10,000 images.

## Data Preprocessing
Fashion-MNIST's original images use the `uint8` data type, with pixel values ranging from `0` to `255`.

The training and test images were converted to `float32`. The pixel values were divided by `255.0`, producing values from `0.0` to `1.0`. The same preprocessing was performed in Go.

## Model Architecture
Defining the Model:

```python
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28), name="image"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="hidden"),
        tf.keras.layers.Dense(10, name="logits"),
    ]
)
```

* Inputs will be 28 by 28 images. There is no batch size specified so that Keras can accept one or more images at once
* The data will be rearranged into one flattened sequence containing 784 values. This does not discard pixels, it only changes the shape
* The Dense layer will connect every one of the 784 input values to each of its 128 neurons. ReLU helps the hidden layer learn nonlinear relationships
* The output layer defines that there are 10 output neurons because Fashion-MNIST has ten categories

Compiling the Model:

```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

* `optimizer="adam"` determines how weights are updated
* `SparseCategoricalCrossentropy` measures classification error. The final ten values are logits, so the loss uses `from_logits=True`
* `metrics=["accuracy"]` reports the percentage of correct predictions

## Training the Model

```python
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
)
```

Training output and trends to look for:
* `loss` - if training loss decreases, the model's training errors are becoming smaller
* `accuracy` - if training accuracy increases, the model is learning the training examples
* `val_loss` - if the validation loss is decreasing, its errors on validation examples are also becoming smaller
* `val_accuracy` - if validation accuracy is increasing, the model's learning transfers to unseen validation examples

## Results

| Epoch | Training accuracy | Training loss | Validation accuracy | Validation loss |
|---:|---:|---:|---:|---:|
| 1 | 82.36% | 0.5048 | 85.30% | 0.4113 |
| 2 | 86.19% | 0.3822 | 86.50% | 0.3673 |
| 3 | 87.61% | 0.3421 | 86.63% | 0.3606 |
| 4 | 88.32% | 0.3175 | 87.30% | 0.3506 |
| 5 | 89.00% | 0.2986 | 87.83% | 0.3407 |

Test Accuracy: 87.14%

Test Loss: 0.3622
### Results Analysis
Training accuracy increased from 82.36% to 89.00% over five epochs, while validation accuracy increased from 85.30% to 87.83%. Training and validation loss also decreased during every epoch, indicating that the model continued learning useful patterns.

The final difference between training and validation accuracy was 1.17 percentage points, showing a small generalization gap and no clear evidence of harmful overfitting. The model achieved 87.14% accuracy on the 10,000 image test set, which was close to its final validation accuracy of 87.83%. This suggests that validation performance was a reasonable estimate of performance on unseen data.

## Python Inference Benchmark

### Methodology

The saved Keras model was loaded from `artifacts/fashion_mnist.keras`. The 10,000 Fashion-MNIST test images were converted to `float32` and
normalized from `0–255` to `0.0–1.0`, matching the preprocessing used during training.

The model performed one untimed warm-up prediction over the complete test set. It then performed five timed inference runs using a batch size
of 32. Each timed run predicted all 10,000 test images. Model loading, data loading, normalization, accuracy calculation, and file writing were excluded from the timed section.

### Results

| Run | Duration (seconds) |
|---:|---:|
| 1 | 0.1307 |
| 2 | 0.1334 |
| 3 | 0.1314 |
| 4 | 0.1327 |
| 5 | 0.1325 |
| **Average** | **0.1321** |

- Test images per run: 10,000
- Batch size: 32
- Average throughput: approximately 75,675 images per second
- Prediction accuracy: 87.14%
- Raw results: `results/python_benchmark.json`
- Saved predictions: `results/python_predictions.csv`

### Interpretation

The five measured durations were close together, ranging from approximately 0.1307 to 0.1334 seconds. This indicates that the full
warm-up prediction reduced one-time initialization effects and produced consistent measurements. The model processed approximately 75,675
images per second in batches of 32.

The benchmark predictions achieved 87.14% accuracy, matching the earlier test evaluation. This confirms that loading the saved model preserved
its learned weights and prediction behavior.

These results establish the Python baseline. The GoMLX benchmark is compared with this Python baseline below.

## ONNX Export and Verification

The trained Keras model was exported to `artifacts/fashion_mnist.onnx` using ONNX opset 15. The exported model accepts a dynamic batch of normalized `float32` images with shape `(batch, 28, 28)` and produces ten logits for each image.

The ONNX checker accepted the exported graph, and ONNX Runtime successfully executed it on all 10,000 test images.

| Measurement | Keras | ONNX |
|---|---:|---:|
| Test accuracy | 87.14% | 87.14% |
| Predicted classes | 10,000 | 10,000 |

The Keras and ONNX models agreed on all 10,000 predicted classes, producing 100% class agreement. The maximum absolute difference between
their raw logits was approximately `1.53 × 10⁻⁵` or `0.00001526`. This small difference is consistent with normal floating-point variation between execution engines and did not change any predicted classes.

These results indicate that the ONNX conversion preserved the model's prediction behavior.

## GoMLX Benchmark 

The trained model was loaded into Go using `artifacts/fashion_mnist.onnx`. The raw Fashion-MNIST test images were converted to `float32` and normalized to values ranging from `0.0-1.0`.

GoMLX used its pure-Go backend to run the model. The reusable executor processed batches of 32 and a final batch of 16. One complete pass was used as an untimed warm-up while five complete passes were timed. Data loading, normalization, model loading, accuracy calculation, class conversion, and file writing were outside the timed section.

These were the GoMLX results for each of the five runs:

| Run | GoMLX duration |
|---:|---:|
| 1 | 0.2068 seconds |
| 2 | 0.2055 seconds |
| 3 | 0.2055 seconds |
| 4 | 0.2086 seconds |
| 5 | 0.2054 seconds |
| **Average** | **0.2064 seconds** |

Raw results: `results/go_benchmark.json`

Predictions: `results/go_predictions.csv`


### Comparison between Python Model and Go Model

| Measurement | Python/TensorFlow | GoMLX |
|---|---:|---:|
| Average duration | 0.1321 s | 0.2064 s |
| Throughput | 75,675 images/s | 48,457 images/s |
| Accuracy | 87.14% | 87.14% |
| Matching predictions | 10,000 | 10,000 |

GoMLX and TensorFlow both achieved 87.14% accuracy and 10,000 matching predictions. This indicates that the ONNX export preserved the model's classification behavior.

In this experiment, TensorFlow achieved 1.56x the throughput of the GoMLX pure-Go backend. GoMLX's average duration was 56.17% longer, and its throughput was 35.97% lower. These results compare TensorFlow with the GoMLX pure-Go backend; they do not demonstrate that Python is inherently faster than Go. Performance may be affected by framework implementation, backend optimization, hardware, and benchmark design.


## Reproduce the Project

Run these commands in your terminal to run both models. The Go program expects the Python export step to have created the ONNX model and binary test-data files first.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements.txt

python python/train.py
python python/benchmark.py
python python/export_onnx.py

deactivate
cd go
go run .
```



## AI Disclaimer

While I wrote and ran the project myself, I used Codex for guidance, explanations, debugging, and code review for:

* setting up, training, and benchmarking the Python model for the Fashion-MNIST dataset
* loading and exporting the Python model using ONNX
* building, testing, and benchmarking the exported model in Go

## Attribution

The Python model architecture was adapted from TensorFlow's
[Basic classification: Classify images of clothing](https://www.tensorflow.org/tutorials/keras/classification)
tutorial. See `THIRD_PARTY_NOTICES.md` for applicable third-party
copyright and license notices.



