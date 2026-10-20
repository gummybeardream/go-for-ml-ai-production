# Repository Guidance

This repository compares two implementations of the same deep-learning
model:

- `python/` contains model development, Python benchmarking, and ONNX export.
- `go/` contains the GoMLX implementation and benchmarking.
- `artifacts/` contains exported models shared between the implementations.
- `results/` contains measurements used in the final analysis.

Keep model inputs, preprocessing, tensor shapes, and output semantics
consistent across the Python and Go implementations.

The root `README.md` is the final assignment write-up, including the
methodology, results, and interpretation.