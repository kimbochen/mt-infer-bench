# Multi-Turn Inference Benchmark

Modified based on vLLM's multi-turn benchmark scripts, with the following differences:

- Loads from actual dataset spec instead of probability distributions
- Definition of a turn: A user-assistant interaction
- Uses a larger corpus for creating requests
- Fixes some benchmarking logic
- Collects metrics from Prometheus endpoints
