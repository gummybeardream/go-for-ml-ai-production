package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gomlx/gomlx/core/tensors"
)

const (
	testImageCount = 10_000
	imageHeight    = 28
	imageWidth     = 28

	imagesPath = "../artifacts/fashion_mnist_test_images.bin"
	labelsPath = "../artifacts/fashion_mnist_test_labels.bin"
)

func readFile(path string, expectedSize int) []byte {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("could not read %s: %v", path, err)
	}

	if len(data) != expectedSize {
		log.Fatalf(
			"%s contains %d bytes; expected %d",
			path,
			len(data),
			expectedSize,
		)
	}

	return data
}

func normalizeImage(imageBytes []byte, imageIndex int) []float32 {
	pixelsPerImage := imageHeight * imageWidth

	start := imageIndex * pixelsPerImage
	end := start + pixelsPerImage

	if imageIndex < 0 || end > len(imageBytes) {
		log.Fatalf("image index %d is out of range", imageIndex)
	}

	normalized := make([]float32, pixelsPerImage)

	for index, pixel := range imageBytes[start:end] {
		normalized[index] = float32(pixel) / 255.0
	}

	return normalized
}

func main() {
	expectedImageBytes := testImageCount * imageHeight * imageWidth

	testImages := readFile(imagesPath, expectedImageBytes)
	testLabels := readFile(labelsPath, testImageCount)

	fmt.Printf("Loaded %d image bytes\n", len(testImages))
	fmt.Printf("Loaded %d labels\n", len(testLabels))

	firstImage := normalizeImage(testImages, 0)

	inputTensor := tensors.FromFlatDataAndDimensions(
		firstImage,
		1,
		imageHeight,
		imageWidth,
	)

	fmt.Printf("Input tensor shape: %s\n", inputTensor.Shape())

	minimum := firstImage[0]
	maximum := firstImage[0]

	for _, pixel := range firstImage {
		if pixel < minimum {
			minimum = pixel
		}

		if pixel > maximum {
			maximum = pixel
		}
	}

	fmt.Printf("First image contains %d normalized pixels\n", len(firstImage))
	fmt.Printf("First image range: %.1f to %.1f\n", minimum, maximum)
	fmt.Printf("First image label: %d\n", testLabels[0])
}
