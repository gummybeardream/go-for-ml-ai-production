package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

const (
	testImageCount = 10_000
	imageHeight    = 28
	imageWidth     = 28

	imagesPath = "../artifacts/fashion_mnist_test_images.bin"
	labelsPath = "../artifacts/fashion_mnist_test_labels.bin"
	modelPath  = "../artifacts/fashion_mnist.onnx"
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

func argmax(values []float32) int {
	largestIndex := 0

	for index := 1; index < len(values); index++ {
		if values[index] > values[largestIndex] {
			largestIndex = index
		}
	}

	return largestIndex
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

	onnxModel, err := parser.ParseFile(modelPath)
	if err != nil {
		log.Fatalf("could not load ONNX model: %v", err)
	}
	defer onnxModel.Close()

	fmt.Printf("Loaded ONNX model: %s\n", onnxModel.Name())

	inputNames, inputShapes := onnxModel.Inputs()

	for index, name := range inputNames {
		fmt.Printf(
			"Model input %d: name=%s shape=%s\n",
			index,
			name,
			inputShapes[index],
		)
	}
	outputNames, outputShapes := onnxModel.Outputs()

	for index, name := range outputNames {
		fmt.Printf(
			"Model output %d: name=%s shape=%s\n",
			index,
			name,
			outputShapes[index],
		)
	}

	backend, err := gobackend.New("")
	if err != nil {
		log.Fatalf("could not create GoMLX backend: %v", err)
	}
	defer backend.Finalize()

	fmt.Printf("GoMLX backend: %s\n", backend.Name())

	// create a model store and load the weights
	store := model.NewStore()
	defer store.Finalize()

	err = onnxModel.VariablesToScope(store.RootScope())
	if err != nil {
		log.Fatalf("could not load ONNX weights: %v", err)
	}

	// execute the model
	outputs, err := model.CallOnceN(
		backend,
		store,
		func(scope *model.Scope, input *graph.Node) []*graph.Node {
			return onnxModel.CallGraph(
				scope,
				input.Graph(),
				map[string]*graph.Node{
					"image": input,
				},
				outputNames...,
			)
		},
		inputTensor,
	)
	if err != nil {
		log.Fatalf("could not run GoMLX inference: %v", err)
	}

	if len(outputs) != 1 {
		log.Fatalf("model produced %d outputs; expected 1", len(outputs))
	}

	outputTensor := outputs[0]
	defer outputTensor.FinalizeAll()

	fmt.Printf("Output tensor shape: %s\n", outputTensor.Shape())

	scores := tensors.MustCopyFlatData[float32](outputTensor)

	if len(scores) != 10 {
		log.Fatalf("model produced %d scores; expected 10", len(scores))
	}

	predictedClass := argmax(scores)

	fmt.Printf("Output scores: %.4f\n", scores)
	fmt.Printf("Predicted class: %d\n", predictedClass)
	fmt.Printf("Actual class: %d\n", testLabels[0])

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
