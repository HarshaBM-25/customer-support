3
"""
clean.py — Phase 2: Text cleaning and light PII scrubbing.

Strips @mentions, URLs, normalizes order/tracking numbers to placeholders.
Keeps raw text alongside cleaned text for debugging.

Usage:
    python -m src.clean
    python -m src.clean --input data/processed/amazon_threads.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm


# ── Cleaning functions ───────────────────────────────────────────────────────

def strip_mentions(text: str) -> str:
    """Remove @username mentions (e.g. @AmazonHelp, @427329)."""
    return re.sub(r"@\w+", "", text)


def strip_urls(text: str) -> str:
    """Remove URLs (http/https links, t.co short links)."""
    return re.sub(r"https?://\S+", "<URL>", text)


def normalize_order_ids(text: str) -> str:
    """Replace Amazon order IDs (format: NNN-NNNNNNN-NNNNNNN) with placeholder.

    Also catches slight variations (with # prefix, spaces, etc).
    """
    # Standard Amazon order ID: 3 digits - 7 digits - 7 digits
    text = re.sub(r"\b\d{3}[-\s]?\d{7}[-\s]?\d{7}\b", "<ORDER_ID>", text)
    return text


def normalize_tracking_numbers(text: str) -> str:
    """Replace long digit sequences (10+ digits) that look like tracking/case numbers."""
    # Only replace standalone long digit sequences (not part of dates, etc.)
    text = re.sub(r"\b\d{10,}\b", "<TRACKING_ID>", text)
    return text


def strip_agent_signoff(text: str) -> str:
    """Remove agent sign-off tags like ^CR, ^TD, ^TN at end of Amazon replies."""
    return re.sub(r"\s*\^[A-Z]{1,3}\s*$", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space, strip edges."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Apply all cleaning steps in order.

    Order matters:
    1. Strip @mentions (before URL removal, since mentions don't contain URLs)
    2. Strip URLs (replace with <URL> placeholder)
    3. Normalize order IDs
    4. Normalize tracking numbers
    5. Strip agent sign-offs
    6. Normalize whitespace
    """
    text = strip_mentions(text)
    text = strip_urls(text)
    text = normalize_order_ids(text)
    text = normalize_tracking_numbers(text)
    text = strip_agent_signoff(text)
    text = normalize_whitespace(text)
    return text


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Clean text in amazon_threads.jsonl")
    p.add_argument("--input", type=str,
                   default="./data/processed/amazon_threads.jsonl",
                   help="Input JSONL file (default: ./data/processed/amazon_threads.jsonl)")
    p.add_argument("--output", type=str,
                   default="./data/processed/amazon_threads_clean.jsonl",
                   help="Output JSONL file (default: ./data/processed/amazon_threads_clean.jsonl)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found at {input_path}")
        print("Run `python -m src.ingest` first to generate the threads file.")
        sys.exit(1)

    # Load threads
    threads = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))

    print(f"Loaded {len(threads):,} threads from {input_path}")

    # Clean each turn's text, keeping raw alongside
    cleaned_count = 0
    for thread in tqdm(threads, desc="Cleaning threads"):
        for turn in thread["turns"]:
            raw = turn["text"]
            cleaned = clean_text(raw)
            turn["text_raw"] = raw       # preserve original
            turn["text"] = cleaned       # overwrite with cleaned
            cleaned_count += 1

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Saved {len(threads):,} threads ({cleaned_count:,} turns cleaned)")
    print(f"    to {output_path}")

    # ── Sanity check: show before/after for a few examples ───────────────
    print("\n" + "=" * 60)
    print("CLEANING EXAMPLES (before → after)")
    print("=" * 60)

    shown = 0
    for thread in threads:
        for turn in thread["turns"]:
            raw = turn["text_raw"]
            cleaned = turn["text"]
            # Only show turns where cleaning actually changed something
            if raw != cleaned and shown < 8:
                print(f"\n  RAW   : {raw[:120]}")
                print(f"  CLEAN : {cleaned[:120]}")
                shown += 1
        if shown >= 8:
            break

    # ── Stats ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLEANING STATS")
    print("=" * 60)
    total_turns = 0
    turns_changed = 0
    mentions_removed = 0
    urls_replaced = 0
    orders_replaced = 0
    tracking_replaced = 0

    for thread in threads:
        for turn in thread["turns"]:
            total_turns += 1
            raw = turn["text_raw"]
            if raw != turn["text"]:
                turns_changed += 1
            mentions_removed += len(re.findall(r"@\w+", raw))
            urls_replaced += len(re.findall(r"https?://\S+", raw))
            orders_replaced += len(re.findall(r"\b\d{3}[-\s]?\d{7}[-\s]?\d{7}\b", raw))
            tracking_replaced += len(re.findall(r"\b\d{10,}\b", raw))

    print(f"  Total turns processed  : {total_turns:,}")
    print(f"  Turns with changes     : {turns_changed:,} ({turns_changed/total_turns*100:.1f}%)")
    print(f"  @mentions removed      : {mentions_removed:,}")
    print(f"  URLs replaced          : {urls_replaced:,}")
    print(f"  Order IDs normalized   : {orders_replaced:,}")
    print(f"  Tracking IDs normalized: {tracking_replaced:,}")


if __name__ == "__main__":
    main()
