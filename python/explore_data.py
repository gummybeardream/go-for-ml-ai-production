import matplotlib.pyplot as plt
import tensorflow as tf

# Identify the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# Download the datasets into specific variables
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Explore the dataset's basic properties
print("Training images shape:", train_images.shape)
print("Training label shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test label shape:", test_labels.shape)
print("Training images pixel type:", train_images.dtype)
print("Training images smallest pixel value:", train_images.min())
print("Training images largest pixel value", train_images.max())

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("Train label name and number:", class_names[train_labels[0]], train_labels[0])

# Display the first 25 images from the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
