# vLLM Multi-Turn Benchmark Design Analysis

Reference: `vllm/benchmarks/multi_turn/`

---

## Program Structure

```
vllm/benchmarks/multi_turn/
├── bench_utils.py                     # Shared utilities (logging, colors)
├── bench_dataset.py                   # Dataset generation & conversation management
├── benchmark_serving_multi_turn.py    # Main benchmark orchestrator
├── convert_sharegpt_to_openai.py      # Standalone converter: ShareGPT → OpenAI format
├── generate_multi_turn.json           # Example config for synthetic generation
└── requirements.txt
```

---

## File-by-File Breakdown

### 1. `bench_utils.py` — Shared Utilities

Minimal file providing:
- **`Color` enum** — ANSI escape codes for colored terminal output (RED, GREEN, BLUE, etc.)
- **`TEXT_SEPARATOR`** — 100-dash string for visual section breaks
- **`logger`** — preconfigured `logging.Logger` (INFO level, timestamped format)

### 2. `bench_dataset.py` — Dataset Generation & Conversation Management

**Type aliases** (lines 17-27):
- `ConvId = str` — conversation ID
- `ShareGptConversations = list[dict[str, Any]]` — list of `{"id": ..., "messages": [...]}`
- `MessagesList = list[dict[str, str]]` — list of `{"role": "user"/"assistant", "content": "..."}`
- `ConversationsMap = dict[ConvId, MessagesList]` — the core data structure mapping IDs → messages

**Distribution classes** (abstract base + 5 implementations):

| Class | Purpose |
|---|---|
| `Distribution` (ABC) | Abstract base; requires `sample(size) → np.ndarray` |
| `UniformDistribution` | Uniform int/float sampling between min/max |
| `ConstantDistribution` | Always returns the same value |
| `ZipfDistribution` | Zipf power-law sampling with optional max cap |
| `PoissonDistribution` | Poisson sampling with optional max cap |
| `LognormalDistribution` | Lognormal sampling; can be parameterized by `average`+`median_ratio` (auto-computes mu/sigma) or raw `mean`+`sigma`. Scales samples to match target average. |

**`GenConvArgs`** (NamedTuple, line 204) — holds all parsed generation parameters: num_conversations, text_files, distributions for turns/tokens/prefixes.

**Key functions:**

- **`get_random_distribution(conf, section, subsection)`** (line 224) — Factory that reads a JSON config block and returns the appropriate `Distribution` subclass based on `"distribution"` field value (`"constant"`, `"zipf"`, `"poisson"`, `"lognormal"`, `"uniform"`).

- **`parse_input_json_file(conf)`** (line 293) — Validates a JSON config dict (must have `filetype: "generate_conversations"`, `num_conversations`, `text_files`, `prompt_input`, `prompt_output`). Calls `get_random_distribution()` for each parameter and returns a `GenConvArgs`.

- **`generate_conversations(args, tokenizer)`** (line 400) — The core synthetic generation engine:
  1. Reads text files and tokenizes them into a flat token pool (`list_of_tokens`)
  2. Samples turn counts per conversation from `input_num_turns` distribution
  3. Samples prefix token counts from `input_prefix_num_tokens` distribution
  4. Optionally generates a **common prefix** (shared across all conversations) from `input_common_prefix_num_tokens`
  5. For each conversation, alternates user/assistant messages:
     - **User turns**: Constructed from `"{conv_id} is a nice number..."` + optional common prefix + optional per-conversation prefix + base prompt text + text sliced from the token pool to hit the target token count
     - **Assistant turns**: Placeholder content (text from token pool); only the token count matters — actual content gets replaced by LLM responses at runtime
  6. Uses `base_offset` to slide through the token pool, reducing text overlap between conversations

- **`conversations_list_to_dict(input_list)`** (line 556) — Converts `ShareGptConversations` (list format: `[{"id": ..., "messages": ...}]`) to `ConversationsMap` (dict format). Prints statistics.

- **`conversations_dict_to_list(input_dict)`** (line 594) — Inverse of above.

- **`print_conv_stats(conversations, tokenizer)`** (line 346) — Tokenizes all messages, computes per-conversation stats (turns, avg user/assistant tokens), and per-request cumulative token counts (simulating chat history growth). Prints percentile summaries.

### 3. `benchmark_serving_multi_turn.py` — Main Benchmark Orchestrator

This is the largest file (~1667 lines). It orchestrates the entire benchmark via multiprocessing.

**Data structures:**

| NamedTuple | Purpose |
|---|---|
| `ClientArgs` | Per-client config: seed, max_requests, skip_first_turn, max_turns, max_active_conversations, verbose, sampling strategy, request_rate, max_retries |
| `RequestArgs` | Per-request config: chat_url, model, stream, min/max token limits, timeout |
| `BenchmarkArgs` | Top-level: url, num_clients, early_stop |
| `ServerResponse` | Raw HTTP response data: valid, ttft_ms, tpot_ms, latency_ms, content, num_chunks |
| `RequestStats` | Processed metrics: ttft, tpot, latency, input/output tokens, cached%, conversation_id, client_id |

**Helper classes:**
- `MetricStats` — tracks min/max/avg/sum/count for a metric
- `MovingAverage` — sliding window average (circular buffer)
- `DebugStats` — aggregates `MetricStats` + `MovingAverage` for ttft, tpot, latency, tokens; prints periodic summaries

**Key functions and the flow:**

#### Entry point: `main()` (line 1271)

1. Parses CLI args (`argparse`)
2. Loads tokenizer (`AutoTokenizer.from_pretrained`)
3. Calls `get_server_info()` — queries `/version` and `/v1/models` endpoints
4. **Reads the input file** — two paths:
   - **If JSON is a list** → treats it as pre-made ShareGPT conversations → calls `conversations_list_to_dict()`
   - **If JSON is a dict with `"filetype"`** → treats it as a generation config → calls `parse_input_json_file()` then `generate_conversations()`
5. Calls `get_client_config()` to build `ClientArgs` and `RequestArgs`
6. Optionally runs a **warmup step** (`--warmup-step`): sends only the first turn of every conversation, results excluded from final stats
7. Runs the **main benchmark** via `main_mp()`
8. Calls `process_statistics()` to compute and print results

#### `main_mp()` (line 883) — Multiprocessing Orchestrator

1. Creates three `mp.Queue`s:
   - **`task_queue`** — input conversations fed to workers
   - **`result_queue`** — `RequestStats` measurements coming back from workers
   - **`conv_queue`** — completed conversations (with LLM answers) coming back
2. Creates an `mp.Event` (`stop_event`) for graceful termination
3. Spawns `num_clients` worker processes, each running `worker_function()` → `client_main()`
4. Enqueues all conversations into `task_queue`, followed by `TERM_SIGNAL` sentinels
5. Main process loop: drains `conv_queue` and `result_queue`, prints progress/stats
6. On early stop (default): once any client finishes, signals all others to stop
7. Joins all worker processes, cleans up queues

#### `client_main()` (line 550) — Per-Client Event Loop

Each client runs in its own process with its own `asyncio` event loop:

1. Seeds RNG uniquely per client
2. Maintains `active_convs` (dict of currently-in-progress conversations) and a `conv_id_queue` (deque for round-robin)
3. Main loop:
   - Pulls new conversations from `task_queue` until it has `max_active_conversations`
   - Picks the next conversation via **round-robin** (pop from deque) or **random** selection
   - Calls `send_turn()` for the current turn
   - On success: increments turn counter, checks if conversation is complete → puts finished conversations on `conv_queue`
   - On failure: retries with `exponential_backoff_sleep()`, removes conversation on final failure
   - Sleeps between requests using `poisson_sleep()` if `request_rate > 0`
4. Sends `TERM_SIGNAL` on `conv_queue` when done

#### `send_turn()` (line 358) — Single Turn Handler

1. Takes `conversation_messages` and `messages_to_use` (how many turns to include)
2. Slices messages to include all history up to the current user turn
3. Resolves `min_tokens`/`max_tokens`: if set to `NUM_TOKENS_FROM_DATASET` (0), reads the expected output length from the dataset's assistant answer
4. Calls `send_request()` to make the HTTP call
5. Post-processes: counts input/output tokens, estimates prefix cache hit rate (`history_tokens / total_input_tokens`), adjusts TTFT/TPOT for multi-token chunks
6. **Updates the conversation in-place**: replaces the placeholder assistant content with the actual LLM response (this is what gets sent as context in subsequent turns)
7. Returns `RequestStats`

#### `send_request()` (line 211) — HTTP Request

1. Builds a JSON payload for `/v1/chat/completions` (OpenAI-compatible API) with model, messages, seed, temperature=0
2. If streaming enabled: iterates `response.content` chunks, parses SSE `data:` lines, extracts `delta.content`, tracks:
   - **TTFT**: time from request start to first content chunk
   - **TPOT**: mean inter-chunk delay (decoding phase)
   - **Latency**: total request time (until `[DONE]`)
3. If non-streaming: reads the full response at once
4. Returns `ServerResponse`

#### `process_statistics()` (line 1072) — Results Processing

1. Converts all `RequestStats` to a pandas DataFrame
2. Optionally computes inter-turn delay per conversation
3. For each warmup percentage: trims leading samples, computes runtime and req/sec, prints percentile summaries (p25, p50, p75, p90, p99, etc.)
4. Optionally exports to Excel

### 4. `convert_sharegpt_to_openai.py` — Dataset Converter (Standalone Tool)

**Purpose**: Converts raw ShareGPT JSON data (HuggingFace format) into OpenAI API format suitable for the benchmarker.

**`convert_sharegpt_to_openai()`** (line 108):
1. Reads ShareGPT JSON (list of `{"id": "convID_part", "conversations": [{"from": "human", "value": "..."}]}`)
2. Merges multi-part conversations by `conv_id`
3. Converts roles: `human/user` → `"user"`, `gpt/bing/chatgpt/bard` → `"assistant"`, `system` → `"system"`
4. Filters: validates turn alternation, content length bounds, English-only text, min/max turn counts
5. Optionally downsamples to `max_items`
6. Writes output as `[{"id": "...", "messages": [{"role": "...", "content": "..."}]}]`

### 5. `generate_multi_turn.json` — Example Config

```json
{
    "filetype": "generate_conversations",
    "num_conversations": 24,
    "text_files": ["pg1184.txt"],
    "prompt_input": {
        "num_turns": {"distribution": "uniform", "min": 12, "max": 18},
        "common_prefix_num_tokens": {"distribution": "constant", "value": 500},
        "prefix_num_tokens": {"distribution": "lognormal", "average": 1000, "max": 5000},
        "num_tokens": {"distribution": "uniform", "min": 120, "max": 160}
    },
    "prompt_output": {
        "num_tokens": {"distribution": "uniform", "min": 80, "max": 120}
    }
}
```

---

## End-to-End Flow: Input File → Benchmark → Request

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Either a ShareGPT JSON list or a generation config  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    main() reads & detects type
                               │
              ┌────────────────┴────────────────┐
              │                                 │
         list (pre-made)              dict (generation config)
              │                                 │
   conversations_list_to_dict()     parse_input_json_file()
              │                     → GenConvArgs
              │                     generate_conversations()
              │                     → tokenize text files
              │                     → sample distributions
              │                     → build synthetic convos
              │                                 │
              └────────────────┬────────────────┘
                               │
                    ConversationsMap (dict)
                    {conv_id: [{role, content}, ...]}
                               │
                    get_client_config()
                    → ClientArgs, RequestArgs
                               │
              ┌────────────────┴────────────────┐
              │ (optional warmup)               │
              │ main_mp() with max_turns=1      │
              │ → sends only first turn         │
              │ → updates assistant content     │
              └────────────────┬────────────────┘
                               │
                    main_mp() — main benchmark
                               │
              ┌────────────────┴────────────────┐
              │  mp.Queue fan-out to N workers  │
              │                                 │
              │  task_queue ──→ client processes │
              │  result_queue ←── RequestStats  │
              │  conv_queue ←── finished convos │
              └────────────────┬────────────────┘
                               │
              Per worker (client_main):
              │
              ├── Pull conversation from task_queue
              ├── Pick next conversation (round_robin/random)
              ├── send_turn():
              │     ├── Slice messages[:current_turn]
              │     ├── Resolve min/max tokens from dataset
              │     ├── send_request():
              │     │     ├── POST /v1/chat/completions
              │     │     ├── Stream SSE chunks
              │     │     ├── Measure TTFT, TPOT, latency
              │     │     └── Return ServerResponse
              │     ├── Count input/output tokens
              │     ├── Adjust TTFT/TPOT for multi-token chunks
              │     ├── Replace placeholder assistant content
              │     │   with actual LLM response (in-place)
              │     └── Return RequestStats → result_queue
              ├── If conversation done → conv_queue
              ├── poisson_sleep() between requests
              └── Loop until no more work
                               │
                    Main process collects all results
                               │
                    process_statistics()
                    → Pandas DataFrame
                    → Percentile summaries
                    → Optional Excel export
                               │
                    Optional: write output JSON
                    (conversations with real LLM answers)
```

**Key design insight**: The assistant messages in the input dataset serve a dual purpose — their **token count** controls `min_tokens`/`max_tokens` for the request (so you get deterministic output lengths), but their **content** is replaced with the actual LLM response after each turn, which then becomes part of the chat history for subsequent turns. This means the benchmark faithfully simulates multi-turn conversations where context grows with each turn.
