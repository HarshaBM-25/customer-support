"""
llm_client.py — Thin wrapper around NVIDIA Nemotron via OpenAI-compatible API.

All LLM calls go through this module so the provider/model can be swapped
with a one-line change.  Every call is logged to data/results/llm_calls.jsonl.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
MODEL = "nvidia/nemotron-3-super-120b-a12b"
BASE_URL = "https://integrate.api.nvidia.com/v1"
LOG_DIR = Path("data/results")
LOG_FILE = LOG_DIR / "llm_calls.jsonl"

# ── Client init ──────────────────────────────────────────────────────────────
_api_key = os.getenv("NVIDIA_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set.  "
        "Copy .env.example → .env and add your key from https://build.nvidia.com/"
    )

_client = OpenAI(base_url=BASE_URL, api_key=_api_key)


# ── Logging helper ───────────────────────────────────────────────────────────
def _log_call(*, system_prompt, user_prompt, model, temperature, raw_response,
              parsed_json=None):
    """Append one record to the LLM call log (JSONL)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": raw_response,
        "parsed_json": parsed_json,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Core call ────────────────────────────────────────────────────────────────
def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Send a single chat-completion request and return the text response.

    Handles rate-limit errors with exponential back-off (up to 3 retries).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(4):  # 0, 1, 2, 3 → up to 3 retries
        try:
            resp = _client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            _log_call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                model=MODEL,
                temperature=temperature,
                raw_response=text,
            )
            return text
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Model retired — not retryable, fail immediately with clear message
            if "410" in err_str or "end of life" in err_str:
                raise RuntimeError(
                    f"Model '{MODEL}' has been retired. Update the MODEL constant "
                    f"in llm_client.py to an available model."
                ) from e
            # Retry on rate-limit or transient server errors
            if "rate" in err_str or "429" in err_str or "500" in err_str or "503" in err_str:
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                print(f"[llm_client] Rate-limited / server error, retrying in {wait}s "
                      f"(attempt {attempt + 1}/3)…")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"LLM call failed after 3 retries: {last_err}") from last_err


# ── JSON helper ──────────────────────────────────────────────────────────────
def _extract_json_obj(text: str) -> dict | list:
    """Extract a JSON object or list from arbitrary text."""
    import re
    cleaned = text.strip()
    # 1. Look for markdown code fence
    fence_match = re.search(r"```(?:json)?\s*([\{\[][\s\S]*?[\}\]])\s*```", cleaned)
    if fence_match:
        return json.loads(fence_match.group(1))

    # 2. Try direct json.loads
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Find first { or [ and scan backwards from the end for matching closing brace
    start_brace = cleaned.find("{")
    start_bracket = cleaned.find("[")
    
    start_idx = -1
    open_char, close_char = "", ""
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
        open_char, close_char = "{", "}"
    elif start_bracket != -1:
        start_idx = start_bracket
        open_char, close_char = "[", "]"

    if start_idx != -1:
        for end_idx in range(len(cleaned), start_idx, -1):
            if cleaned[end_idx - 1] == close_char:
                try:
                    return json.loads(cleaned[start_idx:end_idx])
                except json.JSONDecodeError:
                    continue

    raise json.JSONDecodeError("No valid JSON structure found in response", text, 0)


def call_llm_json(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> dict:
    """Call the LLM with json_mode=True and parse the response.

    Retries up to 3 times on JSON parse failures.
    """
    for attempt in range(3):
        raw = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            parsed = _extract_json_obj(raw)
            if isinstance(parsed, list):
                parsed = {"items": parsed}
            # Re-log with the parsed JSON
            _log_call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                model=MODEL,
                temperature=temperature,
                raw_response=raw,
                parsed_json=parsed,
            )
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[llm_client] JSON parse failed (attempt {attempt + 1}/3): {e}")
            if attempt == 2:
                raise ValueError(
                    f"Could not parse LLM response as JSON after 3 attempts. "
                    f"Last response:\n{raw}"
                ) from e


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Model : {MODEL}")
    print(f"Base   : {BASE_URL}")
    print(f"Log    : {LOG_FILE}")
    print()
    print("Sending test prompt…")
    reply = call_llm("Say hello in one sentence.", temperature=0.5)
    print(f"Reply  : {reply}")
