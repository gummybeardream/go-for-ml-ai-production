package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

const (
	testImageCount       = 10_000
	imageHeight          = 28
	imageWidth           = 28
	inferenceBatchSize   = 32
	classCount           = 10
	benchmarkRepetitions = 5
	warmupRuns           = 1

	imagesPath            = "../artifacts/fashion_mnist_test_images.bin"
	labelsPath            = "../artifacts/fashion_mnist_test_labels.bin"
	modelPath             = "../artifacts/fashion_mnist.onnx"
	goPredictionsPath     = "../results/go_predictions.csv"
	pythonPredictionsPath = "../results/python_predictions.csv"
	goBenchmarkPath       = "../results/go_benchmark.json"
)

type benchmarkResults struct {
	TestImages                int       `json:"test_images"`
	BatchSize                 int       `json:"batch_size"`
	WarmupRuns                int       `json:"warmup_runs"`
	Repetitions               int       `json:"repetitions"`
	DurationsSeconds          []float64 `json:"durations_seconds"`
	AverageDurationSeconds    float64   `json:"average_duration_seconds"`
	ThroughputImagesPerSecond float64   `json:"throughput_images_per_second"`
	Accuracy                  float64   `json:"accuracy"`
	ClassAgreement            float64   `json:"class_agreement"`
	Backend                   string    `json:"backend"`
}

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

func inferBatch(
	inferenceExecutor *model.Exec,
	normalizedImages []float32,
	startImage int,
	numberOfImages int,
) ([]float32, error) {
	pixelsPerImage := imageHeight * imageWidth
	startPixel := startImage * pixelsPerImage
	endPixel := (startImage + numberOfImages) * pixelsPerImage

	if startPixel < 0 || endPixel > len(normalizedImages) {
		return nil, fmt.Errorf(
			"normalized image range %d through %d is out of range",
			startImage,
			startImage+numberOfImages,
		)
	}

	batchImages := normalizedImages[startPixel:endPixel]

	inputTensor := tensors.FromFlatDataAndDimensions(
		batchImages,
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

	return scores, nil
}

func runInference(
	inferenceExecutor *model.Exec,
	normalizedImages []float32,
) ([]float32, error) {
	pixelsPerImage := imageHeight * imageWidth

	if len(normalizedImages)%pixelsPerImage != 0 {
		return nil, fmt.Errorf(
			"normalized data contains %d values; expected a multiple of %d",
			len(normalizedImages),
			pixelsPerImage,
		)
	}

	numberOfImages := len(normalizedImages) / pixelsPerImage
	allScores := make([]float32, 0, numberOfImages*classCount)

	for startImage := 0; startImage < numberOfImages; startImage += inferenceBatchSize {
		currentBatchSize := inferenceBatchSize

		remainingImages := numberOfImages - startImage
		if remainingImages < inferenceBatchSize {
			currentBatchSize = remainingImages
		}

		batchScores, err := inferBatch(
			inferenceExecutor,
			normalizedImages,
			startImage,
			currentBatchSize,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"infer batch starting at image %d: %w",
				startImage,
				err,
			)
		}

		allScores = append(allScores, batchScores...)
	}

	return allScores, nil
}

func scoresToPredictions(scores []float32) ([]int, error) {
	if len(scores)%classCount != 0 {
		return nil, fmt.Errorf(
			"received %d scores; expected a multiple of %d",
			len(scores),
			classCount,
		)
	}

	numberOfImages := len(scores) / classCount
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

func writePredictions(path string, predictions []int) error {
	data := make([]byte, 0, len(predictions)*2)

	for _, prediction := range predictions {
		data = strconv.AppendInt(data, int64(prediction), 10)
		data = append(data, '\n')
	}

	err := os.WriteFile(path, data, 0o644)
	if err != nil {
		return fmt.Errorf("write predictions to %s: %w", path, err)
	}

	return nil
}

func readPredictions(path string) ([]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read predictions from %s: %w", path, err)
	}

	fields := strings.Fields(string(data))
	predictions := make([]int, len(fields))

	for index, field := range fields {
		prediction, err := strconv.Atoi(field)
		if err != nil {
			return nil, fmt.Errorf(
				"parse prediction on line %d: %w",
				index+1,
				err,
			)
		}

		predictions[index] = prediction
	}

	return predictions, nil
}

func writeBenchmarkResults(
	path string,
	results benchmarkResults,
) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("encode benchmark results: %w", err)
	}

	data = append(data, '\n')

	err = os.WriteFile(path, data, 0o644)
	if err != nil {
		return fmt.Errorf(
			"write benchmark results to %s: %w",
			path,
			err,
		)
	}

	return nil
}

func main() {
	expectedImageBytes := testImageCount * imageHeight * imageWidth

	testImages := readFile(imagesPath, expectedImageBytes)
	testLabels := readFile(labelsPath, testImageCount)

	normalizedTestImages := normalizeBatch(
		testImages,
		0,
		testImageCount,
	)

	fmt.Printf(
		"Normalized %d test images before inference\n",
		testImageCount,
	)

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

	// Warm up the executor before collecting measurements.
	_, err = runInference(
		inferenceExecutor,
		normalizedTestImages,
	)
	if err != nil {
		log.Fatalf("could not complete warm-up inference: %v", err)
	}

	fmt.Println("Completed warm-up inference")

	durations := make([]float64, 0, benchmarkRepetitions)
	var allScores []float32

	// Measure five complete inference runs.
	for repetition := 0; repetition < benchmarkRepetitions; repetition++ {
		startTime := time.Now()

		allScores, err = runInference(
			inferenceExecutor,
			normalizedTestImages,
		)

		elapsedTime := time.Since(startTime).Seconds()

		if err != nil {
			log.Fatalf(
				"could not complete benchmark run %d: %v",
				repetition+1,
				err,
			)
		}

		durations = append(durations, elapsedTime)

		fmt.Printf(
			"Run %d: %.6f seconds\n",
			repetition+1,
			elapsedTime,
		)
	}

	// Calculate the average duration and throughput.
	totalDuration := 0.0

	for _, duration := range durations {
		totalDuration += duration
	}

	averageDuration := totalDuration / float64(len(durations))
	throughput := float64(testImageCount) / averageDuration

	fmt.Printf(
		"Average duration: %.6f seconds\n",
		averageDuration,
	)
	fmt.Printf(
		"Throughput: %.2f images per second\n",
		throughput,
	)

	allPredictions, err := scoresToPredictions(allScores)
	if err != nil {
		log.Fatalf("could not convert scores to predictions: %v", err)
	}

	if len(allScores) != testImageCount*classCount {
		log.Fatalf(
			"generated %d scores; expected %d",
			len(allScores),
			testImageCount*classCount,
		)
	}

	if len(allPredictions) != testImageCount {
		log.Fatalf(
			"generated %d predictions; expected %d",
			len(allPredictions),
			testImageCount,
		)
	}

	correctPredictions := 0

	for index, prediction := range allPredictions {
		if prediction == int(testLabels[index]) {
			correctPredictions++
		}
	}

	accuracy := float64(correctPredictions) / float64(testImageCount)

	err = writePredictions(goPredictionsPath, allPredictions)
	if err != nil {
		log.Fatalf("could not save Go predictions: %v", err)
	}

	fmt.Printf(
		"Saved %d Go predictions to %s\n",
		len(allPredictions),
		goPredictionsPath,
	)

	pythonPredictions, err := readPredictions(pythonPredictionsPath)
	if err != nil {
		log.Fatalf("could not read Python predictions: %v", err)
	}

	if len(pythonPredictions) != len(allPredictions) {
		log.Fatalf(
			"prediction counts differ: Python=%d Go=%d",
			len(pythonPredictions),
			len(allPredictions),
		)
	}

	matchingPredictions := 0

	for index, goPrediction := range allPredictions {
		if goPrediction == pythonPredictions[index] {
			matchingPredictions++
		}
	}

	classAgreement := float64(matchingPredictions) /
		float64(len(allPredictions))

	fmt.Printf(
		"Python/Go class agreement: %d/%d = %.2f%%\n",
		matchingPredictions,
		len(allPredictions),
		classAgreement*100,
	)

	fmt.Printf("Generated %d predictions\n", len(allPredictions))
	fmt.Printf(
		"GoMLX accuracy: %d/%d = %.2f%%\n",
		correctPredictions,
		testImageCount,
		accuracy*100,
	)
	results := benchmarkResults{
		TestImages:                testImageCount,
		BatchSize:                 inferenceBatchSize,
		WarmupRuns:                warmupRuns,
		Repetitions:               benchmarkRepetitions,
		DurationsSeconds:          durations,
		AverageDurationSeconds:    averageDuration,
		ThroughputImagesPerSecond: throughput,
		Accuracy:                  accuracy,
		ClassAgreement:            classAgreement,
		Backend:                   backend.Name(),
	}

	err = writeBenchmarkResults(goBenchmarkPath, results)
	if err != nil {
		log.Fatalf("could not save Go benchmark results: %v", err)
	}

	fmt.Printf("Saved benchmark results to %s\n", goBenchmarkPath)
}
