"""Multi-turn benchmark client.

Benchmarks online inference by sending multi-turn sessions to an
OpenAI-compatible chat completions API. Sessions are defined by an
offline spec (exact turn counts and token counts per turn) with text
content sampled from text files.

A "turn" is one user-assistant round (1 user message + 1 assistant response).
"""

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import threading
import time
import urllib.request
from collections import Counter, deque
from datetime import datetime
from enum import Enum
from http import HTTPStatus
from statistics import mean
from typing import NamedTuple

import aiohttp  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from bench_dataset import (
    SessionsMap,
    SessionId,
    MessagesList,
    OfflineSpec,
    ShareGptSessions,
    sessions_dict_to_list,
    generate_sessions_from_spec,
    parse_offline_spec,
    print_session_stats,
)
from bench_utils import TEXT_SEPARATOR, Color, logger
from transformers import AutoTokenizer  # type: ignore

NUM_TOKENS_FROM_DATASET = 0
TERM_SIGNAL = None


class SessionSampling(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"

    def __str__(self):
        return self.value


class ClientArgs(NamedTuple):
    seed: int
    max_num_requests: int | None
    skip_first_round: bool
    max_turns: int | None  # max rounds (user-assistant pairs)
    max_active_sessions: int
    verbose: bool
    print_content: bool
    verify_output: bool
    session_sampling: SessionSampling
    request_rate: float
    max_retries: int


class RequestArgs(NamedTuple):
    chat_url: str
    model: str
    stream: bool
    limit_min_tokens: int  # Use negative value for no limit
    limit_max_tokens: int  # Use negative value for no limit
    timeout_sec: int


class BenchmarkArgs(NamedTuple):
    url: str
    num_clients: int
    early_stop: bool


class ServerResponse(NamedTuple):
    valid: bool
    ttft_ms: float  # time to first chunk
    tpot_ms: float  # time per output chunk (one or more tokens)
    latency_ms: float
    start_time_ms: float
    first_chunk: str  # first chunk of the content
    content: str  # includes the first_chunk
    num_chunks: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}"


class RequestStats(NamedTuple):
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    input_num_turns: int  # round number (1-based)
    input_num_tokens: int
    output_num_tokens: int
    output_num_chunks: int
    output_num_first_chunk_tokens: int
    approx_cached_percent: float
    session_id: str
    client_id: int

    def __str__(self) -> str:
        return (
            f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}, input_num_tokens {self.input_num_tokens}, "
            f"output_num_tokens {self.output_num_tokens} ({self.output_num_chunks} chunks, {self.output_num_first_chunk_tokens} tokens in first chunk), "
            f"approx_cached_percent {self.approx_cached_percent:.2f}%"
        )


class MetricStats:
    def __init__(self) -> None:
        self.min: float | None = None
        self.max: float | None = None
        self.avg: float | None = None
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        if self.min is None:
            self.min = value
        else:
            self.min = min(self.min, value)

        if self.max is None:
            self.max = value
        else:
            self.max = max(self.max, value)

        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f}, min: {self.min:>10.3f}, max: {self.max:>10.3f}"


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.window = np.zeros(window_size)
        self.index = 0
        self.sum = 0.0
        self.count = 0
        self.avg: float | None = None

    def update(self, new_value: float) -> None:
        if self.count < self.window_size:
            self.sum += new_value
            self.window[self.count] = new_value
            self.count += 1
        else:
            old_value = self.window[self.index]
            self.sum = self.sum - old_value + new_value
            self.window[self.index] = new_value
            self.index = (self.index + 1) % self.window_size

        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f} ({self.count} samples)"


class DebugStats:
    def __init__(self, logger: logging.Logger, window_size: int) -> None:
        self.logger = logger
        self.metrics: dict[str, MovingAverage | MetricStats] = {
            "moving_avg_ttft_ms": MovingAverage(window_size),
            "moving_avg_tpot_ms": MovingAverage(window_size),
            "ttft_ms": MetricStats(),
            "tpot_ms": MetricStats(),
            "latency_ms": MetricStats(),
            "input_num_turns": MetricStats(),
            "input_num_tokens": MetricStats(),
            "output_num_tokens": MetricStats(),
        }

    def update(self, data: RequestStats) -> None:
        self.metrics["ttft_ms"].update(data.ttft_ms)
        self.metrics["moving_avg_ttft_ms"].update(data.ttft_ms)
        self.metrics["tpot_ms"].update(data.tpot_ms)
        self.metrics["moving_avg_tpot_ms"].update(data.tpot_ms)
        self.metrics["latency_ms"].update(data.latency_ms)
        self.metrics["input_num_turns"].update(data.input_num_turns)
        self.metrics["input_num_tokens"].update(data.input_num_tokens)
        self.metrics["output_num_tokens"].update(data.output_num_tokens)

    def print(self) -> None:
        self.logger.info("-" * 50)
        for k, v in self.metrics.items():
            kv_info = f"[{k:25}] {v}"
            self.logger.info(kv_info)
        self.logger.info("-" * 50)


class PrometheusCollector:
    """Periodically scrapes /metrics from the server in a background thread."""

    def __init__(
        self, url: str, interval_sec: float = 5.0,
        external_stop=None,
    ) -> None:
        self.metrics_url = f"{url}/metrics"
        self.interval_sec = interval_sec
        self.snapshots: list[dict] = []
        self._stop_event = threading.Event()
        self._external_stop = external_stop
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._scrape_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Prometheus collector started (interval={self.interval_sec}s, "
            f"url={self.metrics_url})"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_sec + 2)
        logger.info(f"Prometheus collector stopped ({len(self.snapshots)} snapshots)")

    def _scrape_loop(self) -> None:
        while not self._stop_event.is_set():
            # Stop scraping when benchmark early-stops
            if self._external_stop is not None and self._external_stop.is_set():
                break
            try:
                self._scrape_once()
            except Exception as e:
                logger.debug(f"Prometheus scrape failed: {e}")
            self._stop_event.wait(self.interval_sec)

    def _scrape_once(self) -> None:
        req = urllib.request.Request(self.metrics_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
        metrics = self._parse_metrics(text)
        self.snapshots.append({"timestamp": time.time(), "metrics": metrics})

    @staticmethod
    def _parse_metrics(text: str) -> dict[str, float]:
        """Parse Prometheus exposition format into {metric_key: value}."""
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    continue
        return metrics


def nanosec_to_millisec(value: float) -> float:
    return value / 1000000.0


def nanosec_to_sec(value: float) -> float:
    return value / 1000000000.0


async def send_request(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    chat_url: str,
    model: str,
    stream: bool = True,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    timeout_sec: int = 120,
) -> ServerResponse:
    payload = {
        "model": model,
        "messages": messages,
        "seed": 0,
        "temperature": 0.0,
    }

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": False}

    if min_tokens is not None:
        payload["min_tokens"] = min_tokens

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {"Content-Type": "application/json"}

    # Calculate the timeout for the request
    if max_tokens is not None:
        # Assume TPOT of 200ms and use max_tokens to determine timeout
        token_based_timeout = int(max_tokens * 0.2)
        if token_based_timeout > timeout_sec:
            timeout_sec = token_based_timeout
            logger.info(
                "Using timeout of %ds based on max_tokens %d",
                timeout_sec,
                max_tokens,
            )
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    valid_response = True
    ttft: float | None = None
    chunk_delay: list[int] = []
    latency: float | None = None
    first_chunk = ""
    generated_text = ""

    start_time: int = time.perf_counter_ns()
    most_recent_timestamp: int = start_time

    async with session.post(
        url=chat_url, json=payload, headers=headers, timeout=timeout
    ) as response:
        http_status = HTTPStatus(response.status)
        if http_status == HTTPStatus.OK:
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                if chunk == "[DONE]":
                    latency = time.perf_counter_ns() - start_time
                elif stream is False:
                    data = json.loads(chunk)
                    message = data["choices"][0]["message"]
                    assert message["role"] == "assistant"
                    generated_text += message["content"]
                else:
                    timestamp: int = time.perf_counter_ns()
                    data = json.loads(chunk)

                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if ttft is None:
                            first_token_time = time.perf_counter_ns()
                            ttft = first_token_time - start_time
                            first_chunk = delta["content"]
                        else:
                            chunk_delay.append(timestamp - most_recent_timestamp)

                        generated_text += delta["content"]

                    most_recent_timestamp = timestamp
        else:
            valid_response = False
            content = await response.text()
            logger.warning(
                f"{Color.YELLOW}Received HTTP status {http_status.value} "
                f"({http_status.phrase}): {content}{Color.RESET}"
            )

    if latency is None:
        latency = -1.0
        if valid_response:
            latency = time.perf_counter_ns() - start_time

    if ttft is None:
        ttft = latency

    tpot: float = mean(chunk_delay) if len(chunk_delay) > 0 else 0.0
    num_chunks: int = len(chunk_delay)

    sr = ServerResponse(
        valid=valid_response,
        ttft_ms=nanosec_to_millisec(ttft) if ttft > 0.0 else -1.0,
        tpot_ms=nanosec_to_millisec(tpot),
        latency_ms=nanosec_to_millisec(latency),
        start_time_ms=nanosec_to_millisec(start_time),
        first_chunk=first_chunk,
        content=generated_text,
        num_chunks=num_chunks,
    )
    return sr


def get_short_string(input: str) -> str:
    n = 20
    if len(input) < 400:
        return input
    return f"{input[:n]}...{input[-n:]}"


def get_token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def get_messages_token_count(
    tokenizer: AutoTokenizer, messages: list[dict[str, str]]
) -> int:
    token_count = 0
    for m in messages:
        token_count += get_token_count(tokenizer, m["content"])
    return token_count


async def send_turn(
    session: aiohttp.ClientSession,
    client_id: int,
    session_id: str,
    session_messages: MessagesList,
    messages_to_use: int,
    tokenizer: AutoTokenizer,
    req_args: RequestArgs,
    verbose: bool,
    verify_output: bool,
) -> RequestStats | None:
    assert messages_to_use > 0
    assert messages_to_use <= len(session_messages)

    messages = session_messages[:messages_to_use]

    # Index of the current message (should be "user")
    index = messages_to_use - 1

    assert len(messages[index].keys()) == 2
    assert "role" in messages[index] and "content" in messages[index]
    assert messages[index]["role"] == "user", (
        f"Failed on session ID {session_id}, message role should be user"
    )

    if verbose:
        print(
            f"{Color.CYAN}Messages (session ID {session_id},"
            f" {len(messages)} messages, turn {(messages_to_use + 1) // 2}):{Color.RESET}",
            messages,
        )

    # None means no upper/lower limit for the output token count
    min_tokens = None if req_args.limit_min_tokens < 0 else req_args.limit_min_tokens
    max_tokens = None if req_args.limit_max_tokens < 0 else req_args.limit_max_tokens

    if len(session_messages) > messages_to_use:
        # The session contains an assistant answer for the next user prompt
        if (
            min_tokens == NUM_TOKENS_FROM_DATASET
            or max_tokens == NUM_TOKENS_FROM_DATASET
        ):
            # Compute number of tokens in the answer (from the input session)
            assistant_answer = session_messages[messages_to_use]
            answer_num_tokens = get_token_count(tokenizer, assistant_answer["content"])
            assert assistant_answer["role"] == "assistant"

        if min_tokens == NUM_TOKENS_FROM_DATASET:
            min_tokens = max(1, answer_num_tokens)

        if max_tokens == NUM_TOKENS_FROM_DATASET:
            max_tokens = max(1, answer_num_tokens)

    # Send the current session to the LLM and get a response
    response: ServerResponse = await send_request(
        session,
        messages,
        req_args.chat_url,
        req_args.model,
        req_args.stream,
        min_tokens,
        max_tokens,
        req_args.timeout_sec,
    )

    if response.valid is False:
        return None

    # Compute number of tokens in input / output
    input_num_tokens = get_messages_token_count(tokenizer, messages)

    # Num tokens in the user's last question
    question_num_tokens = get_token_count(tokenizer, messages[index]["content"])

    # Num tokens in the history/context of the question
    assert input_num_tokens >= question_num_tokens
    history_num_tokens = input_num_tokens - question_num_tokens

    # Num tokens in the LLM's answer (first chunk and full answer)
    first_chunk_tokens = get_token_count(tokenizer, response.first_chunk)

    output_content = response.content
    output_num_tokens = get_token_count(tokenizer, output_content)

    # Prefix caching approximated cached percent
    approx_cached_percent = (
        100.0 * (history_num_tokens / input_num_tokens) if input_num_tokens > 0 else 0.0
    )

    # Compute the correct TTFT and TPOT (based on tokens and not chunks).
    # Required because multiple output tokens may be bundled in a single chunk.
    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
        decode_ms = response.latency_ms - response.ttft_ms
        decode_num_tokens = output_num_tokens - first_chunk_tokens
        tpot_ms = decode_ms / decode_num_tokens
    else:
        tpot_ms = 0.0

    if first_chunk_tokens > 1:
        delta_ms = (first_chunk_tokens - 1) * tpot_ms
        ttft_ms = max(0.1, response.ttft_ms - delta_ms)
    else:
        ttft_ms = response.ttft_ms

    # Turn number (1-based): which user-assistant round this request represents
    turn_number = (messages_to_use + 1) // 2

    rs = RequestStats(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=response.latency_ms,
        start_time_ms=response.start_time_ms,
        input_num_turns=turn_number,
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        output_num_chunks=response.num_chunks,
        output_num_first_chunk_tokens=first_chunk_tokens,
        approx_cached_percent=approx_cached_percent,
        session_id=session_id,
        client_id=client_id,
    )

    if verbose:
        print(
            f"\n{Color.YELLOW}Response ({output_num_tokens} tokens):{Color.RESET}",
            output_content,
        )
        print(f"{Color.YELLOW}Response metrics: {rs}{Color.RESET}")
        print("-" * 70)

    # Save the LLM's answer (used as context for the next user turn)
    answer_index = messages_to_use
    if len(session_messages) > answer_index:
        assert session_messages[answer_index]["role"] == "assistant", (
            f"Failed on session ID {session_id}, message role should be assistant"
        )

        orig_content = session_messages[answer_index]["content"]
        if verify_output:
            debug_info = (
                f"LLM/dataset answers do not match ({session_id}):"
                f"\n'{get_short_string(output_content)}' (len: {len(output_content)}),"
                f"\n'{get_short_string(orig_content)}' (len: {len(orig_content)})"
            )
            if orig_content != output_content:
                raise ValueError(debug_info)

        # Update the answer
        session_messages[answer_index]["content"] = output_content
    else:
        # A user prompt that has no answer, add the answer as a new message
        new_answer = {"role": "assistant", "content": output_content}
        session_messages.append(new_answer)

    return rs


async def poisson_sleep(request_rate: float, verbose: bool = False) -> None:
    assert request_rate > 0
    interval = np.random.exponential(1.0 / request_rate)
    if verbose:
        logger.info(f"Sleeping for {interval:.3f} seconds...")
    await asyncio.sleep(interval)


async def exponential_backoff_sleep(
    attempt_cnt: int,
    base_rate: float = 1.0,
    backoff_factor: float = 2.0,
    jitter_fraction: float = 0.10,
    verbose: bool = False,
) -> None:
    backoff_delay = base_rate * (backoff_factor**attempt_cnt)
    jittered_delay = backoff_delay * (
        1 + np.random.uniform(-jitter_fraction, jitter_fraction)
    )
    if verbose:
        logger.info(f"Backoff for {jittered_delay:.3f} seconds...")
    await asyncio.sleep(jittered_delay)


async def client_main(
    args: ClientArgs,
    req_args: RequestArgs,
    client_id: int,
    tokenizer: AutoTokenizer,
    stop_event: mp.Event,  # type: ignore
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    session_queue: mp.Queue,
) -> None:
    logger.info(
        f"{Color.CYAN}Started client {client_id}: "
        f"max_num_requests={args.max_num_requests}, "
        f"max_active_sessions={args.max_active_sessions}{Color.RESET}"
    )

    # Set unique seed per client (each client runs in its own process)
    client_seed = args.seed + client_id + 1
    random.seed(client_seed)
    np.random.seed(client_seed)

    # Active sessions
    active_sessions: SessionsMap = {}
    session_id_queue: deque = deque(maxlen=args.max_active_sessions)

    # Track completed rounds per session (0-based)
    rounds_done: Counter = Counter()
    num_successes = 0
    num_failures = 0

    # Track the timestamp of the last turn per session (debug only)
    time_of_last_turn: dict[SessionId, float] = {}

    # Flag: no new tasks (sessions) available from the queue
    task_queue_empty = False

    async with aiohttp.ClientSession() as session:
        # Continue while there is work: either tasks in queue or active sessions
        while not task_queue_empty or len(active_sessions) > 0:
            result = None

            if (
                args.max_num_requests
                and num_successes + num_failures == args.max_num_requests
            ):
                logger.info(
                    f"{Color.YELLOW}Client {client_id} reached "
                    f"request limit{Color.RESET}"
                )
                break

            if stop_event.is_set():  # type: ignore
                logger.info(
                    f"{Color.YELLOW}Client {client_id} received "
                    f"a termination signal{Color.RESET}"
                )
                break

            while (
                len(active_sessions) < args.max_active_sessions
                and task_queue_empty is False
            ):
                # Get a new session from the task queue
                session_id, messages = task_queue.get()

                if session_id is TERM_SIGNAL:
                    task_queue_empty = True
                    break

                if args.skip_first_round:
                    # Skip the first round (user + assistant),
                    # relevant if warmup was enabled.
                    rounds_done[session_id] += 1

                total_rounds = len(messages) // 2
                if rounds_done[session_id] < total_rounds:
                    # Add new session
                    active_sessions[session_id] = messages
                    session_id_queue.append(session_id)

                    if args.verbose:
                        logger.info(
                            f"{Color.GREEN}Client {client_id} will use session "
                            f"ID {session_id} (active sessions "
                            f"{len(active_sessions)}){Color.RESET}"
                        )

                elif args.verbose:
                    logger.info(
                        f"{Color.YELLOW}Client {client_id} will not use session "
                        f"ID {session_id} (all {total_rounds} turns already "
                        f"sent){Color.RESET}"
                    )

            if len(active_sessions) == 0 and task_queue_empty:
                logger.info(
                    f"{Color.YELLOW}Client {client_id} has no more work{Color.RESET}"
                )
                break

            # Pick an active session for the next request
            if args.session_sampling == SessionSampling.ROUND_ROBIN:
                session_id = session_id_queue.pop()
            else:
                active_ids = list(active_sessions.keys())
                session_id = random.choice(active_ids)

            messages = active_sessions[session_id]
            assert isinstance(messages, list) and len(messages) > 0

            # Compute the message index for this round
            current_round = rounds_done[session_id]
            messages_to_use = 2 * current_round + 1

            assert messages_to_use < len(messages), (
                f"Round {current_round} is invalid for session ID {session_id}"
                f" that has only {len(messages) // 2} turns"
            )

            if args.verbose:
                curr_time_sec: float = time.perf_counter()
                time_since_last_turn: str | float = "N/A"
                if session_id in time_of_last_turn:
                    time_since_last_turn = round(
                        curr_time_sec - time_of_last_turn[session_id], 3
                    )
                logger.info(
                    f"Client {client_id} using session ID {session_id} "
                    f"(turn: {current_round + 1}, time since last turn [sec]: "
                    f"{time_since_last_turn})"
                )
                time_of_last_turn[session_id] = curr_time_sec

            success = False
            for attempt_cnt in range(args.max_retries + 1):
                try:
                    exception = False
                    result = await send_turn(
                        session,
                        client_id,
                        session_id,
                        messages,
                        messages_to_use,
                        tokenizer,
                        req_args,
                        args.print_content,
                        args.verify_output,
                    )
                    if result is not None:
                        result_queue.put(result)
                        success = True
                        break
                    else:
                        logger.warning(
                            f"{Color.YELLOW}Client {client_id} - Request rejected "
                            f"during session ID {session_id} "
                            f"(turn: {current_round + 1}){Color.RESET}"
                        )
                except asyncio.exceptions.TimeoutError:
                    exception = True
                    logger.error(
                        "%sClient %d - Timeout during session ID %s (turn: %d). "
                        "Base timeout is %ss (set with --request-timeout-sec), but the "
                        "effective timeout may be longer based on max_tokens. If this "
                        "is unexpected, consider increasing the timeout or checking "
                        "model performance.%s",
                        Color.RED,
                        client_id,
                        session_id,
                        current_round + 1,
                        req_args.timeout_sec,
                        Color.RESET,
                    )
                except Exception:
                    exception = True
                    logger.exception(
                        f"{Color.RED}Client {client_id} - Exception during "
                        f"session ID {session_id} "
                        f"(turn: {current_round + 1}){Color.RESET}"
                    )

                # Sleep before retry if not last attempt
                if not success and attempt_cnt < args.max_retries:
                    await exponential_backoff_sleep(attempt_cnt, verbose=args.verbose)

            if not success:
                num_failures += 1
                # Remove the failed session and continue with the next one
                active_sessions.pop(session_id)
                logger.warning(
                    f"{Color.YELLOW}Client {client_id} - Skipping session "
                    f"{session_id} after {args.max_retries + 1} attempts{Color.RESET}"
                )

            else:
                num_successes += 1

                # Round complete (user message sent + assistant response received)
                rounds_done[session_id] += 1

                total_rounds = len(messages) // 2
                if args.max_turns is not None:
                    total_rounds = min(args.max_turns, total_rounds)

                if rounds_done[session_id] >= total_rounds:
                    # Session complete — send only ID (not messages) to avoid
                    # blocking on large serialized data in the pipe buffer
                    active_sessions.pop(session_id)
                    session_queue.put((session_id, None))
                    if args.verbose:
                        logger.info(
                            f"{Color.GREEN}Client {client_id} finished "
                            f"session ID {session_id}{Color.RESET}"
                        )
                else:
                    # Session continues, insert at the back of the queue
                    session_id_queue.appendleft(session_id)

            # Sleep between requests (if rate is positive)
            if args.request_rate > 0:
                await poisson_sleep(args.request_rate, args.verbose)

    # Send indication that the client is done
    session_queue.put((TERM_SIGNAL, TERM_SIGNAL))

    logger.info(
        f"{Color.CYAN}Client {client_id} is done "
        f"({num_successes=}, {num_failures=}){Color.RESET}"
    )


def worker_function(
    client_id: int,
    tokenizer: AutoTokenizer,
    client_args: ClientArgs,
    req_args: RequestArgs,
    stop_event: mp.Event,  # type: ignore
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    session_queue: mp.Queue,
) -> None:
    asyncio.run(
        client_main(
            client_args,
            req_args,
            client_id,
            tokenizer,
            stop_event,
            task_queue,
            result_queue,
            session_queue,
        )
    )


def get_client_config(
    args: argparse.Namespace, input_sessions: SessionsMap
) -> tuple[ClientArgs, RequestArgs]:
    if args.num_clients < 1:
        raise ValueError("Number of clients must be a positive number")

    if len(input_sessions) < args.num_clients:
        raise ValueError(
            "Number of sessions must be equal or larger than the number of clients"
        )

    max_req_per_client: int | None = None
    if args.max_num_requests is not None:
        req_per_client = args.max_num_requests // args.num_clients
        if req_per_client < 1:
            raise ValueError("Number of requests should be at least one per client")
        max_req_per_client = req_per_client

    max_active_sessions = args.max_active_sessions
    if max_active_sessions is None:
        max_active_sessions = args.num_clients

    if max_active_sessions > len(input_sessions):
        raise ValueError(
            f"Max active sessions {max_active_sessions} "
            "must be equal or less than the total number of sessions"
        )

    max_active_conv_per_client = max_active_sessions // args.num_clients
    if max_active_conv_per_client < 1:
        raise ValueError(
            f"Max active sessions {max_active_sessions} "
            "must be equal or greater than the number of clients"
        )

    skip_first_round = args.warmup_step

    client_args = ClientArgs(
        seed=args.seed,
        max_num_requests=max_req_per_client,
        skip_first_round=skip_first_round,
        max_turns=args.max_turns,
        max_active_sessions=max_active_conv_per_client,
        verbose=args.verbose,
        print_content=args.print_content,
        verify_output=args.verify_output,
        session_sampling=args.session_sampling,
        request_rate=args.request_rate,
        max_retries=args.max_retries,
    )

    if args.limit_min_tokens > 0 or args.limit_max_tokens > 0:
        if args.limit_min_tokens < 1 or args.limit_max_tokens < 1:
            raise ValueError(
                "Invalid min/max tokens limits (both limits should be provided)"
            )
        if args.limit_min_tokens > args.limit_max_tokens:
            raise ValueError(
                "Invalid min/max tokens limits (min should not be larger than max)"
            )

    if args.request_timeout_sec <= 0:
        raise ValueError("Request timeout must be a positive number")

    chat_url = f"{args.url}/v1/chat/completions"
    model_name = args.served_model_name if args.served_model_name else args.model

    req_args = RequestArgs(
        chat_url=chat_url,
        model=model_name,
        stream=not args.no_stream,
        limit_min_tokens=args.limit_min_tokens,
        limit_max_tokens=args.limit_max_tokens,
        timeout_sec=args.request_timeout_sec,
    )

    return client_args, req_args


async def main_mp(
    client_args: ClientArgs,
    req_args: RequestArgs,
    bench_args: BenchmarkArgs,
    tokenizer: AutoTokenizer,
    input_sessions: SessionsMap,
    stop_event=None,
) -> tuple[SessionsMap, list[RequestStats]]:
    # An event that will trigger graceful termination of all the clients
    if stop_event is None:
        stop_event = mp.Event()

    num_clients = bench_args.num_clients

    # Balanced session dispatch: sort by total chars (proxy for tokens),
    # round-robin assign to clients, shuffle within each client
    session_items = list(input_sessions.items())
    # Sort by total turns (= total requests). Each turn incurs a fixed
    # sleep overhead, so total turns determines wall time per client.
    session_items.sort(
        key=lambda x: len(x[1]) // 2, reverse=True
    )

    client_assignments: list[list[tuple[str, MessagesList]]] = [
        [] for _ in range(num_clients)
    ]
    for i, item in enumerate(session_items):
        client_assignments[i % num_clients].append(item)

    for assignment in client_assignments:
        random.shuffle(assignment)

    logger.info(
        f"Balanced dispatch: {len(session_items)} sessions across "
        f"{num_clients} clients "
        f"(min {min(len(a) for a in client_assignments)}, "
        f"max {max(len(a) for a in client_assignments)} per client)"
    )

    # Per-client task queues
    task_queues: list[mp.Queue] = []
    for assignment in client_assignments:
        q: mp.Queue = mp.Queue()
        for session_id, messages in assignment:
            q.put((session_id, messages))
        q.put((TERM_SIGNAL, TERM_SIGNAL))
        task_queues.append(q)

    # Queue for client measurements (TTFT, TPOT, etc. for each request)
    result_queue: mp.Queue = mp.Queue()

    # Queue for output sessions (with the LLM answers)
    session_queue: mp.Queue = mp.Queue()
    output_sessions: SessionsMap = {}
    client_metrics: list[RequestStats] = []

    # Start all clients
    start_time = time.perf_counter_ns()
    logger.info(f"{Color.GREEN}Starting {num_clients} clients{Color.RESET}")

    clients = []
    for client_id in range(num_clients):
        client = mp.Process(
            name=f"client_{client_id}",
            target=worker_function,
            args=(
                client_id,
                tokenizer,
                client_args,
                req_args,
                stop_event,
                task_queues[client_id],
                result_queue,
                session_queue,
            ),
        )
        clients.append(client)
        client.start()

    # Collect the updated sessions from all clients
    num_clients_finished = 0
    total_sessions = len(input_sessions)

    debug_stats = DebugStats(logger, min(15 * bench_args.num_clients, 500))

    import queue as _queue

    def _drain_result_queue():
        """Non-blocking drain of result_queue."""
        while True:
            try:
                new_data = result_queue.get_nowait()
                client_metrics.append(new_data)
                debug_stats.update(new_data)
            except _queue.Empty:
                break

    while num_clients_finished < bench_args.num_clients:
        _drain_result_queue()

        # Use timeout to periodically drain result_queue even when
        # session_queue has no new items
        try:
            session_id, messages = session_queue.get(timeout=1.0)
        except _queue.Empty:
            continue

        _drain_result_queue()

        if session_id is TERM_SIGNAL:
            num_clients_finished += 1
            logger.info(
                f"{Color.CYAN}{num_clients_finished} out of "
                f"{bench_args.num_clients} clients finished{Color.RESET}"
            )

            if bench_args.early_stop and not stop_event.is_set():
                logger.info(
                    f"{Color.YELLOW}Sending termination signal to clients{Color.RESET}"
                )
                stop_event.set()
        else:
            output_sessions[session_id] = messages  # None (data stays in client)

            finished_sessions = len(output_sessions)
            percent = finished_sessions / total_sessions

            print_cycle = max(3, int(bench_args.num_clients / 4))

            if finished_sessions % print_cycle == 0:
                runtime_sec = nanosec_to_sec(time.perf_counter_ns() - start_time)
                logger.info(
                    f"{Color.CYAN}Finished {finished_sessions} out of {total_sessions} "
                    f"sessions ({percent:.0%}), "
                    f"{num_clients_finished} out of {bench_args.num_clients} clients "
                    f"finished, collected {len(client_metrics)} measurements, "
                    f"runtime {runtime_sec:.3f} sec{Color.RESET}"
                )

                rps: str | float = round(len(client_metrics) / runtime_sec, 3)
                if len(client_metrics) < (5 * bench_args.num_clients):
                    rps = "N/A"

                runtime_left_sec: str | float = round(
                    (runtime_sec / finished_sessions) * (total_sessions - finished_sessions), 3
                )
                if percent < 0.05:
                    runtime_left_sec = "N/A"

                logger.info(
                    f"{Color.CYAN}Estimated req/sec {rps}, estimated runtime left "
                    f"{runtime_left_sec} sec{Color.RESET}"
                )
                debug_stats.print()

    logger.info(
        f"{Color.CYAN}All {bench_args.num_clients} clients finished{Color.RESET}"
    )

    # Collect remaining results from all the clients
    while not result_queue.empty():
        client_metrics.append(result_queue.get())

    logger.info(f"Collected {len(client_metrics)} samples from all the clients")

    # Wait for all clients to finish
    for client in clients:
        logger.info(
            f"{Color.CYAN}Waiting for client {client.name} "
            f"(is alive: {client.is_alive()}){Color.RESET}"
        )

        client.join(timeout=req_args.timeout_sec + 1)

        if client.is_alive():
            logger.warning(
                f"{Color.YELLOW}Client {client.name} will be terminated{Color.RESET}"
            )
            client.terminate()

        exitcode = client.exitcode
        if exitcode != 0:
            logger.error(
                f"{Color.RED}Client {client.name} exited "
                f"with exit code {exitcode}{Color.RESET}"
            )

    logger.info(
        f"All {bench_args.num_clients} clients exited (successfully "
        f"finished {len(output_sessions)} out of {total_sessions} sessions)"
    )

    # Clean up per-client task queues. Use cancel_join_thread() to avoid
    # hanging on feeder threads that can't flush to dead client processes.
    for tq in task_queues:
        tq.cancel_join_thread()
        tq.close()

    result_queue.close()
    result_queue.join_thread()

    session_queue.close()
    session_queue.join_thread()

    return output_sessions, client_metrics


def get_filename_with_timestamp(label: str, extension: str) -> str:
    time_now = datetime.now()
    timestamp = time_now.strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"{label}__{timestamp}.{extension}"
    return filename


def process_statistics(
    client_metrics: list[RequestStats],
    warmup_percentages: list[float],
    test_params: dict,
    verbose: bool,
    json_output: str | None = None,
    warmup_runtime_sec: float | None = None,
    prometheus_snapshots: list[dict] | None = None,
) -> None:
    if len(client_metrics) == 0:
        logger.info("No samples to process")
        return

    logger.info(f"Processing {len(client_metrics)} samples...")

    raw_data = pd.DataFrame(client_metrics)

    if verbose:
        raw_data = raw_data.sort_values(by=["session_id", "start_time_ms"])
        raw_data["time_between_user_turns_sec"] = raw_data.groupby("session_id")[
            "start_time_ms"
        ].diff()
        raw_data["time_between_user_turns_sec"] = (
            raw_data["time_between_user_turns_sec"] / 1000.0
        )

    # Final raw data should be sorted by time
    raw_data = raw_data.sort_values(by=["start_time_ms"])
    raw_data["end_time_ms"] = raw_data["start_time_ms"] + raw_data["latency_ms"]

    percentiles = [0.25, 0.5, 0.75, 0.9]

    if len(raw_data) >= 100:
        percentiles.append(0.99)

    if len(raw_data) >= 1000:
        percentiles.append(0.999)

    if len(raw_data) >= 10000:
        percentiles.append(0.9999)

    # Set precision for numbers in the output text (the dataframes)
    pd.set_option("display.precision", 2)
    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)

    # Exclude parameters from RequestStats
    exclude = [
        "start_time_ms",
        "end_time_ms",
        "output_num_first_chunk_tokens",
        "approx_cached_percent",
        "session_id",
        "client_id",
    ]

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Parameters:{Color.RESET}")
    for k, v in test_params.items():
        print(f"{k}={v}")
    print(TEXT_SEPARATOR)

    params_list = []
    df_list = []
    for percent in warmup_percentages:
        warmup_count = int(percent * len(raw_data))
        tail_count = len(raw_data) - warmup_count
        if tail_count == 0:
            break

        df = raw_data.tail(tail_count)

        runtime_sec = df["end_time_ms"].iloc[-1] - df["start_time_ms"].iloc[0]
        runtime_sec = runtime_sec / 1000.0
        requests_per_sec = float(len(df)) / runtime_sec
        params = {
            "runtime_sec": runtime_sec,
            "requests_per_sec": requests_per_sec,
        }
        if warmup_runtime_sec is not None:
            params["warmup_runtime_sec"] = warmup_runtime_sec
            params["total_runtime_incl_warmup_sec"] = runtime_sec + warmup_runtime_sec

        df = df.drop(columns=exclude).describe(percentiles=percentiles).transpose()

        params_list.append(params)
        df_list.append(df)

        if percent > 0 or len(warmup_percentages) > 1:
            print(
                f"{Color.YELLOW}Statistics summary "
                f"(assuming {percent:.0%} warmup samples):{Color.RESET}"
            )
        else:
            print(f"{Color.YELLOW}Statistics summary:{Color.RESET}")

        for k, v in params.items():
            if isinstance(v, float):
                print(f"{k} = {v:.3f}")
            else:
                print(f"{k} = {v}")
        print(TEXT_SEPARATOR)
        print(df)
        print(TEXT_SEPARATOR)

    if json_output:
        summaries = []
        for percent, params, df_stats in zip(
            warmup_percentages, params_list, df_list
        ):
            summaries.append({
                "warmup_percent": percent,
                **params,
                "num_samples": int(df_stats.iloc[0]["count"]),
                "statistics": {
                    metric: {k: float(v) for k, v in row.items()}
                    for metric, row in df_stats.iterrows()
                },
            })

        output = {
            "params": test_params,
            "summaries": summaries,
            "raw_requests": [m._asdict() for m in client_metrics],
        }
        if prometheus_snapshots is not None:
            output["prometheus"] = prometheus_snapshots

        with open(json_output, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(
            f"{Color.GREEN}Metrics exported to: {json_output}{Color.RESET}"
        )


async def get_server_info(url: str) -> None:
    logger.info(f"{Color.BLUE}Collecting information from server: {url}{Color.RESET}")
    async with aiohttp.ClientSession() as session:
        url_version = f"{url}/version"
        async with session.get(url_version) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{Color.BLUE}Server version: {text}{Color.RESET}")

        url_models = f"{url}/v1/models"
        async with session.get(url_models) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{Color.BLUE}Models:{Color.RESET}")
                models_data = json.loads(text)
                models_list = models_data["data"]
                for model in models_list:
                    model_id = model["id"]
                    max_model_len = model.get("max_model_len", "N/A")
                    logger.info(
                        f"{Color.BLUE}\t{model_id=}, {max_model_len=}{Color.RESET}"
                    )
            else:
                logger.info(f"{Color.RED}Failed to get models{Color.RESET}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Multi-turn benchmark client",
        description="Benchmark online inference using multi-turn sessions "
        "defined by an offline spec",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input JSON file with offline spec defining sessions "
        "(turn counts and token counts per turn)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file containing sessions with updated assistant answers",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generators (default: 0)",
    )

    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path of the LLM model"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the --model argument.",
    )

    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the LLM API server",
    )

    parser.add_argument(
        "-p",
        "--num-clients",
        type=int,
        default=1,
        help="Number of clients that will send requests in parallel",
    )
    parser.add_argument(
        "-k",
        "--max-active-sessions",
        type=int,
        default=None,
        help="Max number of active sessions at a time (for all clients)",
    )
    parser.add_argument(
        "-n",
        "--max-num-requests",
        type=int,
        default=None,
        help="Max number of requests to send (total for all clients)",
    )

    parser.add_argument(
        "--warmup-step",
        default=False,
        action="store_true",
        help="Run a warmup step (send only the first turn of every session); "
        "measurements will not be included in the final benchmark results",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns per session "
        "(a turn is one user-assistant round), disabled by default",
    )
    parser.add_argument(
        "--no-early-stop",
        default=False,
        action="store_true",
        help="By default, the benchmark stops if at least one client exits. "
        "Use this flag to disable this behavior",
    )

    parser.add_argument(
        "--limit-max-tokens",
        type=int,
        default=NUM_TOKENS_FROM_DATASET,
        help="Set max_tokens for each request output. "
        "Overrides output token count from the spec. "
        "Use a negative value to disable this limit.",
    )
    parser.add_argument(
        "--limit-min-tokens",
        type=int,
        default=NUM_TOKENS_FROM_DATASET,
        help="Set min_tokens for each request output. "
        "Overrides output token count from the spec. "
        "Use a negative value to disable this limit.",
    )

    parser.add_argument(
        "--request-rate",
        type=float,
        default=0,
        help="Expected request rate (Poisson process) per client in requests/sec. "
        "Set to 0 for no delay between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MULTITURN_BENCH_MAX_RETRIES", "0")),
        help="Maximum number of retry attempts for failed requests. "
        "Default is 0 (no retries).",
    )
    parser.add_argument(
        "--session-sampling",
        type=SessionSampling,
        choices=list(SessionSampling),
        default=SessionSampling.ROUND_ROBIN,
        help="Strategy for selecting which session to use for the next request.",
    )
    parser.add_argument(
        "--verify-output",
        default=False,
        action="store_true",
        help="Verify the LLM output (compare to the answers in the input spec)",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=120,
        help="Timeout in seconds for each API request (default: 120). "
        "Automatically increased if max tokens imply longer decoding.",
    )

    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Use at most this many sessions from the spec. "
        "If the spec has more, only the first N are used. "
        "Useful when the spec is sized for the largest sweep point.",
    )

    parser.add_argument(
        "--no-stream",
        default=False,
        action="store_true",
        help="Disable streaming mode (set 'stream' to False in the API request)",
    )

    parser.add_argument(
        "--print-stats",
        default=False,
        action="store_true",
        help="Print session statistics after generation",
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Export metrics (params, summary stats, raw requests, prometheus) "
        "to a JSON file",
    )
    parser.add_argument(
        "--prometheus-interval",
        type=float,
        default=5.0,
        help="Scrape server /metrics endpoint every N seconds during the "
        "benchmark. Set to 0 to disable. (default: 5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--print-content",
        default=False,
        action="store_true",
        help="Print the user prompts and the server's answers",
    )

    parser.add_argument(
        "--warmup-percentages",
        type=str,
        default="0%",
        help="Ignore the first X samples as warmup (X is a percentage). "
        "A comma separated list of percentages can be used "
        "(for example: --warmup-percentages=0%%,50%%)",
    )

    args = parser.parse_args()

    logger.info(args)

    logger.info(f"{Color.GREEN}Input parameters:{Color.RESET}")
    logger.info(f"url={args.url}")
    logger.info(f"model={args.model}")
    logger.info(f"num_clients={args.num_clients}")

    if args.verify_output:
        logger.info(f"{Color.PURPLE}Verify is enabled{Color.RESET}")

    # Calculate warmup percentages
    try:
        warmup_percentages: list[float] = [0.0]
        if not args.warmup_step:
            warmup_strings: list[str] = args.warmup_percentages.split(",")
            warmup_strings = [x.replace("%", "") for x in warmup_strings]
            warmup_percentages = [float(x) / 100 for x in warmup_strings]

            for p in warmup_percentages:
                assert p >= 0.0 and p < 1.0

            warmup_percentages.sort()

            logger.info(
                f"Warmup percentages (percentage of samples): {warmup_percentages}"
            )

    except Exception:
        raise ValueError(
            f"Invalid --warmup-percentages={args.warmup_percentages}"
        ) from None

    # Set global seeds for main process
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    await get_server_info(args.url)

    # Load offline spec
    logger.info(f"Reading input file: {args.input_file}")
    with open(args.input_file) as f:
        input_data = json.load(f)

    assert isinstance(input_data, dict), (
        f"Input file {args.input_file} must be a JSON object (offline spec)"
    )

    spec: OfflineSpec = parse_offline_spec(input_data)
    logger.info(
        f"Loaded spec with {len(spec.sessions)} sessions, "
        f"text files: {spec.text_files}"
    )

    # Disable warning from "huggingface/tokenizers"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Generate sessions from spec (possibly truncated)
    if args.max_sessions is not None:
        if args.max_sessions < 1:
            raise ValueError("--max-sessions must be a positive number")
        if args.max_sessions < len(spec.sessions):
            spec = spec._replace(
                sessions=spec.sessions[: args.max_sessions]
            )
            logger.info(
                f"Truncated spec to {len(spec.sessions)} sessions "
                f"(--max-sessions={args.max_sessions})"
            )

    sessions = generate_sessions_from_spec(spec, tokenizer)

    if args.print_stats:
        print_session_stats(sessions, tokenizer)

    if args.max_turns is not None:
        if args.max_turns < 1:
            raise ValueError("Max turns must be a positive number")
        logger.info(
            f"{Color.PURPLE}Max turns per session "
            f"is limited to {args.max_turns}{Color.RESET}"
        )

    # Create benchmark configurations
    client_args, req_args = get_client_config(args, sessions)

    bench_args = BenchmarkArgs(
        url=args.url, num_clients=args.num_clients, early_stop=not args.no_early_stop
    )

    warmup_runtime_sec: float | None = None

    # Warm-up step
    if args.warmup_step:
        warmup_client_args = client_args._replace(
            skip_first_round=False, max_turns=1, max_active_sessions=1
        )
        warmup_bench_args = bench_args._replace(early_stop=False)

        logger.info("%sWarmup start%s", Color.PURPLE, Color.RESET)
        warmup_start_ns = time.perf_counter_ns()
        sessions, _ = await main_mp(
            warmup_client_args, req_args, warmup_bench_args, tokenizer, sessions
        )
        warmup_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - warmup_start_ns)
        logger.info(
            "%sWarmup runtime: %.3f sec (%.3f ms)%s",
            Color.PURPLE,
            warmup_runtime_sec,
            warmup_runtime_sec * 1000,
            Color.RESET,
        )
        logger.info("%sWarmup done%s", Color.PURPLE, Color.RESET)

    # Shared stop event — used by early stop and prometheus collector
    bench_stop_event = mp.Event()

    # Start prometheus collection
    prom_collector: PrometheusCollector | None = None
    if args.prometheus_interval > 0:
        prom_collector = PrometheusCollector(
            args.url, args.prometheus_interval, external_stop=bench_stop_event
        )
        prom_collector.start()

    # Run the benchmark
    benchmark_start_ns = time.perf_counter_ns()
    client_sessions, client_metrics = await main_mp(
        client_args, req_args, bench_args, tokenizer, sessions,
        stop_event=bench_stop_event,
    )
    benchmark_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - benchmark_start_ns)

    # Stop prometheus collection
    if prom_collector is not None:
        prom_collector.stop()

    requests_per_sec = len(client_metrics) / benchmark_runtime_sec
    benchmark_runtime_ms = benchmark_runtime_sec * 1000.0
    logger.info(
        "%sAll clients finished, benchmark runtime: %.3f sec (%.3f ms), "
        "requests per second: %.3f%s",
        Color.GREEN,
        benchmark_runtime_sec,
        benchmark_runtime_ms,
        requests_per_sec,
        Color.RESET,
    )
    if warmup_runtime_sec is not None:
        total_runtime_sec = benchmark_runtime_sec + warmup_runtime_sec
        logger.info(
            "%sWarmup runtime: %.3f sec (%.3f ms)%s",
            Color.GREEN,
            warmup_runtime_sec,
            warmup_runtime_sec * 1000,
            Color.RESET,
        )
        logger.info(
            "%sTotal runtime (including warmup): %.3f sec (%.3f ms)%s",
            Color.GREEN,
            total_runtime_sec,
            total_runtime_sec * 1000,
            Color.RESET,
        )

    # Benchmark parameters
    params = {
        "model": args.model,
        "num_clients": args.num_clients,
        "num_sessions": len(sessions),
        "active_sessions": args.max_active_sessions,
        "seed": args.seed,
        "text_files": ", ".join(spec.text_files),
    }

    if args.limit_min_tokens > 0:
        params["min_tokens"] = args.limit_min_tokens

    if args.limit_max_tokens > 0:
        params["max_tokens"] = args.limit_max_tokens

    # Process and print statistics
    process_statistics(
        client_metrics,
        test_params=params,
        warmup_percentages=warmup_percentages,
        verbose=args.verbose,
        json_output=args.json_output,
        warmup_runtime_sec=warmup_runtime_sec,
        prometheus_snapshots=(
            prom_collector.snapshots if prom_collector is not None else None
        ),
    )

    if args.output_file is not None:
        output_data: ShareGptSessions = sessions_dict_to_list(client_sessions)
        logger.info(
            f"{Color.GREEN}Writing sessions file: {args.output_file}{Color.RESET}"
        )
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
