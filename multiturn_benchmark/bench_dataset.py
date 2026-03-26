"""Dataset generation from offline specifications.

An offline spec defines the exact shape of each session (number of turns,
input/output token counts per turn). The actual text content is sampled from
text files with wrapping when the token pool is exhausted.

Spec format (JSON):
{
    "text_files": ["pg1184.txt"],
    "conversations": [
        {"num_turns": 3, "input_tokens": [100, 150, 200], "output_tokens": [80, 120, 90]},
        ...
    ]
}

A "turn" is one user-assistant round (1 user message + 1 assistant response).
"""

from statistics import mean
from typing import Any, NamedTuple

import pandas as pd  # type: ignore
from bench_utils import (
    TEXT_SEPARATOR,
    Color,
    logger,
)
from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore

# Session ID is a string (e.g: "SESSION_0")
SessionId = str

# A list of dicts (dicts with keys "id" and "messages")
ShareGptSessions = list[dict[str, Any]]

# A list of dicts (dicts with keys "role" and "content")
MessagesList = list[dict[str, str]]

# Map session ID to session messages
SessionsMap = dict[SessionId, MessagesList]

# Backward-compatible aliases
ConvId = SessionId
ShareGptConversations = ShareGptSessions
ConversationsMap = SessionsMap


class SessionSpec(NamedTuple):
    """Specification for a single session."""

    num_turns: int  # Number of user-assistant rounds
    input_tokens: list[int]  # Token count for each user turn
    output_tokens: list[int]  # Token count for each assistant turn


# Backward-compatible alias
ConvSpec = SessionSpec


class OfflineSpec(NamedTuple):
    """Offline benchmark specification."""

    text_files: list[str]
    sessions: list[SessionSpec]


def parse_offline_spec(conf: dict) -> OfflineSpec:
    """Parse and validate an offline specification dict."""
    assert isinstance(conf, dict), "Spec must be a JSON object"

    text_files = conf.get("text_files")
    assert isinstance(text_files, list) and len(text_files) > 0, (
        "text_files must be a non-empty list"
    )

    raw_sessions = conf.get("sessions")
    assert isinstance(raw_sessions, list) and len(raw_sessions) > 0, (
        "sessions must be a non-empty list"
    )

    sessions = []
    for i, c in enumerate(raw_sessions):
        num_turns = c.get("num_turns")
        assert isinstance(num_turns, int) and num_turns > 0, (
            f"session {i}: num_turns must be a positive int"
        )

        input_tokens = c.get("input_tokens")
        assert isinstance(input_tokens, list) and len(input_tokens) == num_turns, (
            f"session {i}: input_tokens must have length {num_turns}"
        )
        assert all(isinstance(t, int) and t > 0 for t in input_tokens), (
            f"session {i}: all input_tokens must be positive ints"
        )

        output_tokens = c.get("output_tokens")
        assert isinstance(output_tokens, list) and len(output_tokens) == num_turns, (
            f"session {i}: output_tokens must have length {num_turns}"
        )
        assert all(isinstance(t, int) and t > 0 for t in output_tokens), (
            f"session {i}: all output_tokens must be positive ints"
        )

        sessions.append(SessionSpec(num_turns, input_tokens, output_tokens))

    return OfflineSpec(text_files=text_files, sessions=sessions)


def sample_tokens_with_wrap(
    tokens: list[int], offset: int, count: int
) -> list[int]:
    """Sample `count` tokens starting at `offset`, wrapping around if needed."""
    n = len(tokens)
    assert n > 0, "Token pool is empty"

    if count <= 0:
        return []

    offset = offset % n

    if offset + count <= n:
        return tokens[offset : offset + count]

    # Need to wrap around
    result = list(tokens[offset:])
    remaining = count - len(result)
    full_wraps = remaining // n
    for _ in range(full_wraps):
        result.extend(tokens)
    partial = remaining % n
    if partial > 0:
        result.extend(tokens[:partial])

    return result


def generate_sessions_from_spec(
    spec: OfflineSpec, tokenizer: AutoTokenizer
) -> SessionsMap:
    """Generate sessions from an offline spec by sampling text with wrapping."""
    base_prompt_text = "Please rewrite the following text and add more content: "
    base_prompt_token_count = len(
        tokenizer.encode(base_prompt_text, add_special_tokens=False)
    )

    logger.info(
        f"{Color.PURPLE}Generating sessions from offline spec...{Color.RESET}"
    )
    logger.info(f"Number of sessions: {len(spec.sessions)}")

    # Load text files into a single token pool
    list_of_tokens: list[int] = []
    for filename in spec.text_files:
        with open(filename) as f:
            data = f.read()
            tokens_in_file = tokenizer.encode(data, add_special_tokens=False)
            list_of_tokens.extend(tokens_in_file)
        logger.info(
            f"Loaded {len(tokens_in_file)} tokens from {filename}, "
            f"total: {len(list_of_tokens)}"
        )

    assert len(list_of_tokens) > 0, "No tokens loaded from text files"

    sessions: SessionsMap = {}
    offset = 0

    for session_idx, session_spec in enumerate(
        tqdm(
            spec.sessions,
            total=len(spec.sessions),
            desc="Generating sessions",
            unit="session",
        )
    ):
        session_id = f"SESSION_{session_idx}"
        messages: MessagesList = []

        for turn_idx in range(session_spec.num_turns):
            # --- User message ---
            target_input = session_spec.input_tokens[turn_idx]

            # Build preamble (unique per session to avoid shared prefix)
            content = f"{session_idx} is a nice number... "
            content += base_prompt_text
            preamble_tokens = len(
                tokenizer.encode(content, add_special_tokens=False)
            )

            fill_tokens = max(0, target_input - preamble_tokens)
            if fill_tokens > 0:
                sampled = sample_tokens_with_wrap(list_of_tokens, offset, fill_tokens)
                content += tokenizer.decode(sampled)
                offset += fill_tokens

            messages.append({"role": "user", "content": content})

            # --- Assistant placeholder ---
            # Content is only used to determine min_tokens/max_tokens for the
            # API request. Actual content gets replaced by the LLM's response.
            target_output = max(1, session_spec.output_tokens[turn_idx])
            sampled = sample_tokens_with_wrap(list_of_tokens, 0, target_output)
            asst_content = tokenizer.decode(sampled)
            messages.append({"role": "assistant", "content": asst_content})

        sessions[session_id] = messages

        # Advance offset to reduce text overlap between sessions
        offset += session_spec.num_turns

    return sessions


# Backward-compatible alias
generate_conversations_from_spec = generate_sessions_from_spec


def print_session_stats(
    sessions: SessionsMap, tokenizer: AutoTokenizer
) -> None:
    """Print statistics about the generated sessions."""
    session_stats: list[dict[Any, Any]] = []
    req_stats: list[int] = []

    print("\nCollecting statistics...")
    for messages in sessions.values():
        user_tokens: list[int] = []
        assistant_tokens: list[int] = []
        request_tokens: list[int] = []

        req_tokens = 0
        for m in messages:
            content = m["content"]
            num_tokens = len(tokenizer(content).input_ids)

            if m["role"] == "user":
                user_tokens.append(num_tokens)
                req_tokens += num_tokens
                request_tokens.append(req_tokens)
            elif m["role"] == "assistant":
                assistant_tokens.append(num_tokens)
                req_tokens += num_tokens

        item_stats = {
            "num_turns": len(messages) // 2,
            "user_tokens": mean(user_tokens),
            "assistant_tokens": mean(assistant_tokens),
        }

        session_stats.append(item_stats)
        req_stats.extend(request_tokens)

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]

    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)

    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Session statistics:{Color.RESET}")
    print(TEXT_SEPARATOR)
    df = pd.DataFrame(session_stats)
    print(df.describe(percentiles=percentiles).transpose())
    print(TEXT_SEPARATOR)
    print(f"{Color.YELLOW}Request statistics:{Color.RESET}")
    print(TEXT_SEPARATOR)
    df = pd.DataFrame(req_stats, columns=["request_tokens"])
    print(df.describe(percentiles=percentiles).transpose())
    print(TEXT_SEPARATOR)


# Backward-compatible alias
print_conv_stats = print_session_stats


def sessions_dict_to_list(input_dict: SessionsMap) -> ShareGptSessions:
    """Convert SessionsMap to ShareGPT list format."""
    output: ShareGptSessions = []
    for session_id, session_data in input_dict.items():
        output.append({"id": session_id, "messages": session_data})
    return output


# Backward-compatible alias
conversations_dict_to_list = sessions_dict_to_list
