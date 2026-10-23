import tensorflow as tf

# Identify and load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def print_image_statistics(name, images):
    print(f"{name} data type:", images.dtype)
    print(f"{name} shape:", images.shape)
    print(f"{name} minimum pixel value:", images.min())
    print(f"{name} maximum pixel value:", images.max())


print_image_statistics("Training images before normalization", train_images)
print_image_statistics("Test images before normalization", test_images)

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0


def assert_normalized_images(name, images, labels):
    assert images.dtype == "float32", f"{name} must use float32"
    assert images.min() >= 0.0, f"{name} contains a value below 0.0"
    assert images.max() <= 1.0, f"{name} contains a value above 1.0"
    assert images.ndim == 3, f"{name} must have three dimensions"
    assert images.shape[1:] == (28, 28), f"{name} images must be 28 by 28"
    assert images.shape[0] == labels.shape[0], f"{name} must have one label per image"


assert_normalized_images("Training images", train_images, train_labels)
assert_normalized_images("Test images", test_images, test_labels)

print_image_statistics("Training images after normalization", train_images)
print_image_statistics("Test images after normalization", test_images)


# Define and build the model by setting up the layers
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28), name="image"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="hidden"),
        tf.keras.layers.Dense(10, name="logits"),
    ]
)

model.summary()

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Display the training configuration for verification
print("Model compiled:", model.compiled)
print("Optimizer:", type(model.optimizer).__name__)
print("Loss:", type(model.loss).__name__)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
)

# Evaluate the trained model
test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose=2,
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

model.save("artifacts/fashion_mnist.keras")
