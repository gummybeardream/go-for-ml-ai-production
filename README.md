# go-for-ml-ai-production

# Fashion MNIST Python and GoMLX Comparison

## Objective
Compare model execution times under Python and Go using the Fashion-MNIST dataset. Export the Python model using ONNX and read the model in to GOMLX to compare predictions between both models.

After training the model on 60,000 images supplied by the Fashion-MNIST dataset, 10,000 test images will be used to evaluate how well the model can classify each article of clothing into the 10 available numeric labels corresponding to clothing labels.

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


