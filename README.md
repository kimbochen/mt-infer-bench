# Multi-Turn Inference Benchmark

Modified based on vLLM's multi-turn benchmark scripts, with the following differences:

- Loads from actual dataset spec instead of probability distributions
- Definition of a turn: A user-assistant interaction
- Uses a larger corpus for creating requests
- Fixes benchmarking logic to ensure every client finishes all its in-progress conversations before exiting
- Collects metrics from Prometheus endpoints

![](visualization/slo_frontier.png)
