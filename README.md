# go-for-ml-ai-production

# Fashion MNIST Python and GoMLX Comparison

## Objective
Compare model execution times under Python and Go using the Fashion-MNIST dataset. Export the Python model using ONNX and read the model into GoMLX to compare predictions between both models.

The model will be trained on 54,000 images while 6,000 will be reserved for validation for a total of 60,000 training images provided by the Fashion-MNIST dataset. The final 10,000 test images will be used to evaluate how well the model can classify each article of clothing into the 10 available numeric labels corresponding to clothing labels.

The model architecture will be adapted from TensorFlow's Fashion-MNIST classification tutorial. The model will be trained and benchmarked locally for this project.

## Dataset
Fashion-MNIST contains 70,000 grayscale images labeled under 10 numeric labels which correspond to ten clothing categories:

* Images use grayscale pixel value from `0` to `255`
* The original image data type is `uint8`
* Each image has 28 by 28 pixels or contains 784 pixels total

The training set has 60,000 images and the testing set has 10,000 images.

## Data Preprocessing
Fashion-MNIST's original images use `uint8` as their datatype while the original values ranging from `0` to `255`. 

The training and test images have been converted to `float32`. The pixel values have been divided by `255.0`, producing values from `0.0` to `1.0`. The same preprocessing will be performed in Go. 

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
Training accuracy increased from 82.36% to 89.00% over five epochs, while validation accuracy increased from 85.30% to 87.83%. Training and validation loss also decreased during every epoch, indicating that the model continued learning useful patterns. The final difference between training and validation accuracy was 1.17 percentage points, showing a small generalization gap and no clear evidence of harmful overfitting. The model achieved 87.14% accuracy on the 10,000 image test set, which was close to its final validation accuracy of 87.83%. This suggests that validation performance was a reasonable estimate of performance on unseen data.