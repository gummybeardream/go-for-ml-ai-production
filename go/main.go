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
	testImageCount     = 10_000
	imageHeight        = 28
	imageWidth         = 28
	inferenceBatchSize = 32
	classCount         = 10

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

func normalizeBatch(
	imageBytes []byte,
	startImage int,
	numberOfImages int,
) []float32 {
	pixelsPerImage := imageHeight * imageWidth
	totalImages := len(imageBytes) / pixelsPerImage

	if startImage < 0 ||
		numberOfImages <= 0 ||
		startImage+numberOfImages > totalImages {
		log.Fatalf(
			"image range %d through %d is out of range",
			startImage,
			startImage+numberOfImages,
		)
	}

	startByte := startImage * pixelsPerImage
	endByte := (startImage + numberOfImages) * pixelsPerImage

	normalized := make([]float32, numberOfImages*pixelsPerImage)

	for index, pixel := range imageBytes[startByte:endByte] {
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

func predictBatch(
	inferenceExecutor *model.Exec,
	imageBytes []byte,
	startImage int,
	numberOfImages int,
) ([]int, error) {
	normalizedImages := normalizeBatch(
		imageBytes,
		startImage,
		numberOfImages,
	)

	inputTensor := tensors.FromFlatDataAndDimensions(
		normalizedImages,
		numberOfImages,
		imageHeight,
		imageWidth,
	)
	defer inputTensor.FinalizeAll()

	outputs, err := inferenceExecutor.Call(inputTensor)
	if err != nil {
		return nil, fmt.Errorf("run inference: %w", err)
	}

	if len(outputs) != 1 {
		return nil, fmt.Errorf(
			"model produced %d outputs; expected 1",
			len(outputs),
		)
	}

	outputTensor := outputs[0]
	defer outputTensor.FinalizeAll()

	scores, err := tensors.CopyFlatData[float32](outputTensor)
	if err != nil {
		return nil, fmt.Errorf("read output scores: %w", err)
	}

	expectedScoreCount := numberOfImages * classCount

	if len(scores) != expectedScoreCount {
		return nil, fmt.Errorf(
			"model produced %d scores; expected %d",
			len(scores),
			expectedScoreCount,
		)
	}

	predictions := make([]int, numberOfImages)

	for imageIndex := 0; imageIndex < numberOfImages; imageIndex++ {
		scoreStart := imageIndex * classCount
		scoreEnd := scoreStart + classCount

		predictions[imageIndex] = argmax(
			scores[scoreStart:scoreEnd],
		)
	}

	return predictions, nil
}

func main() {
	expectedImageBytes := testImageCount * imageHeight * imageWidth

	testImages := readFile(imagesPath, expectedImageBytes)
	testLabels := readFile(labelsPath, testImageCount)

	fmt.Printf("Loaded %d image bytes\n", len(testImages))
	fmt.Printf("Loaded %d labels\n", len(testLabels))

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

	// Build a reusable inference executor.
	inferenceExecutor, err := model.NewExec(
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
	)
	if err != nil {
		log.Fatalf("could not create inference executor: %v", err)
	}
	defer inferenceExecutor.Finalize()

	//Run inference on the complete test set
	allPredictions := make([]int, 0, testImageCount)
	correctPredictions := 0

	for startImage := 0; startImage < testImageCount; startImage += inferenceBatchSize {
		numberOfImages := inferenceBatchSize

		remainingImages := testImageCount - startImage
		if remainingImages < inferenceBatchSize {
			numberOfImages = remainingImages
		}

		batchPredictions, err := predictBatch(
			inferenceExecutor,
			testImages,
			startImage,
			numberOfImages,
		)
		if err != nil {
			log.Fatalf(
				"could not predict batch starting at image %d: %v",
				startImage,
				err,
			)
		}

		allPredictions = append(allPredictions, batchPredictions...)

		for batchIndex, prediction := range batchPredictions {
			testIndex := startImage + batchIndex

			if prediction == int(testLabels[testIndex]) {
				correctPredictions++
			}
		}
	}

	accuracy := float64(correctPredictions) / float64(testImageCount)

	fmt.Printf("Generated %d predictions\n", len(allPredictions))
	fmt.Printf(
		"GoMLX accuracy: %d/%d = %.2f%%\n",
		correctPredictions,
		testImageCount,
		accuracy*100,
	)
}
