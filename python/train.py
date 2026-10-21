import tensorflow as tf

# Identify and load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def print_image_statistics(name, images):
    print(f"{name} data type:", images.dtype)
    print(f"{name} minimum pixel value:", images.min())
    print(f"{name} maximum pixel value:", images.max())


print_image_statistics("Training images before normalization", train_images)
print_image_statistics("Test images before normalization", test_images)

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0


def assert_normalized_images(name, images):
    assert images.dtype == "float32", f"{name} must use float32"
    assert images.min() >= 0.0, f"{name} contains a value below 0.0"
    assert images.max() <= 1.0, f"{name} contains a value above 1.0"


assert_normalized_images("Training images", train_images)
assert_normalized_images("Test images", test_images)

print_image_statistics("Training images after normalization", train_images)
print_image_statistics("Test images after normalization", test_images)
