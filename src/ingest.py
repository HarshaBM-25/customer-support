"""
ingest.py — Phase 1: Load the Customer Support on Twitter dataset, filter to
AmazonHelp-related conversations, reconstruct multi-turn threads, sample,
and save to data/processed/amazon_threads.jsonl.

Usage:
    python -m src.ingest                         # defaults
    python -m src.ingest --max-threads 10000     # larger sample
    python -m src.ingest --data-dir ./my_data    # custom CSV location
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import pandas as pd
from tqdm import tqdm


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest twcs.csv, filter to AmazonHelp, reconstruct threads."
    )
    p.add_argument("--data-dir", type=str, default="./dataset",
                   help="Directory containing twcs.csv (default: ./dataset)")
    p.add_argument("--max-threads", type=int, default=5000,
                   help="Max threads to keep after sampling (default: 5000)")
    p.add_argument("--output-dir", type=str, default="./data/processed",
                   help="Where to write amazon_threads.jsonl (default: ./data/processed)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling (default: 42)")
    return p.parse_args()


# ── Load CSV ─────────────────────────────────────────────────────────────────
def load_csv(data_dir: str) -> pd.DataFrame:
    """Load twcs.csv with only the columns we need."""
    csv_path = Path(data_dir) / "twcs.csv"
    if not csv_path.exists():
        print(
            f"ERROR: Dataset file not found at {csv_path}\n\n"
            "Please download the 'Customer Support on Twitter' dataset from:\n"
            "  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n\n"
            f"Place twcs.csv in the '{data_dir}/' directory and re-run."
        )
        sys.exit(1)

    cols = [
        "tweet_id", "author_id", "inbound", "created_at",
        "text", "response_tweet_id", "in_response_to_tweet_id",
    ]
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path, usecols=cols)
    print(f"  → {len(df):,} total rows loaded.")
    return df


def _normalize_id(val) -> str | None:
    """Convert a tweet ID (int, float, or string) to a clean string, or None.

    Handles:  3 → "3",  3.0 → "3",  "3.0" → "3",  NaN → None,  "" → None
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("", "nan", "None"):
        return None
    # "3.0" → "3"  (float-encoded ints)
    if "." in s:
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            return s
    return s


def _split_ids(val) -> list[str]:
    """Split a possibly comma-separated ID field like '5,7' into ['5','7'].

    Returns empty list for NaN / empty.
    """
    if pd.isna(val):
        return []
    parts = str(val).strip().split(",")
    result = []
    for p in parts:
        nid = _normalize_id(p.strip())
        if nid:
            result.append(nid)
    return result


# ── Filter to AmazonHelp-related rows ───────────────────────────────────────
def filter_amazon(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that are part of an AmazonHelp conversation.

    Criteria:
    - Rows authored by AmazonHelp  (Amazon's outbound replies)
    - Inbound rows that AmazonHelp replied to  (the customer messages Amazon
      actually engaged with)
    - Any ancestor inbound messages in the same thread chain
    """
    print("Normalizing IDs …")
    # Normalize tweet_id to clean string (handles int→str, float→str)
    df["tweet_id"] = df["tweet_id"].apply(_normalize_id)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].apply(_normalize_id)
    # response_tweet_id can be comma-separated — we'll parse it later
    # Keep inbound as boolean
    df["inbound"] = df["inbound"].fillna(False).astype(bool)

    # Step 1: All AmazonHelp-authored tweets
    amazon_mask = df["author_id"] == "AmazonHelp"
    amazon_tweet_ids = set(df.loc[amazon_mask, "tweet_id"].dropna())

    # Step 2: Tweets that AmazonHelp replied to (parents of Amazon tweets)
    amazon_parents = set(
        df.loc[amazon_mask, "in_response_to_tweet_id"].dropna()
    )
    relevant_ids = amazon_tweet_ids | amazon_parents

    # Step 3: Walk up parent chains to capture full conversation context.
    # Build a lookup: tweet_id → in_response_to_tweet_id
    parent_map = {}
    for _, row in df.iterrows():
        tid = row["tweet_id"]
        parent = row["in_response_to_tweet_id"]
        if tid and parent:
            parent_map[tid] = parent

    # Walk parents of already-relevant tweets
    frontier = set(relevant_ids)
    while frontier:
        new_parents = set()
        for tid in frontier:
            parent = parent_map.get(tid)
            if parent and parent not in relevant_ids:
                new_parents.add(parent)
        relevant_ids |= new_parents
        frontier = new_parents

    # Also walk children: for relevant tweets, pull in their response_tweet_ids
    # Build child lookup from response_tweet_id (comma-separated)
    child_map = defaultdict(list)
    for _, row in df.iterrows():
        tid = row["tweet_id"]
        if tid:
            for child_id in _split_ids(row.get("response_tweet_id")):
                child_map[tid].append(child_id)

    frontier = set(relevant_ids)
    while frontier:
        new_children = set()
        for tid in frontier:
            for child_id in child_map.get(tid, []):
                if child_id not in relevant_ids:
                    new_children.add(child_id)
        relevant_ids |= new_children
        frontier = new_children

    filtered = df[df["tweet_id"].isin(relevant_ids)].copy()
    print(f"  → {len(filtered):,} AmazonHelp-related rows after filtering "
          f"(from {len(df):,}).")
    return filtered


# ── Reconstruct threads ─────────────────────────────────────────────────────
def reconstruct_threads(df: pd.DataFrame) -> list[dict]:
    """Build linear conversation threads from parent→child relationships.

    Returns a list of thread dicts, each with:
      - thread_id:  tweet_id of the root message
      - turns:      list of {tweet_id, author_id, text, created_at, inbound}
      - num_turns, has_amazon_reply, has_followup
    """
    # Build lookups
    tweet_lookup = {}
    children = defaultdict(list)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing tweets"):
        tid = row["tweet_id"]
        if not tid:
            continue
        tweet_lookup[tid] = row

        # Parent→child from in_response_to_tweet_id
        parent = row["in_response_to_tweet_id"]
        if parent:
            children[parent].append(tid)

        # Also use response_tweet_id (comma-separated) for child links
        for child_id in _split_ids(row.get("response_tweet_id")):
            # Add reverse link: this tweet is parent of child_id
            if child_id not in children.get(tid, []):
                children[tid].append(child_id)

    all_tweet_ids = set(tweet_lookup.keys())

    # Find root tweets: tweets whose parent is not in our filtered dataset
    roots = []
    for tid in all_tweet_ids:
        parent = tweet_lookup[tid]["in_response_to_tweet_id"]
        if parent is None or parent not in all_tweet_ids:
            roots.append(tid)

    print(f"  → Found {len(roots):,} root tweets.")

    # Walk each root's thread (BFS to maintain chronological order)
    threads = []
    for root_id in tqdm(roots, desc="Building threads"):
        turns = []
        queue = [root_id]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited or current not in tweet_lookup:
                continue
            visited.add(current)
            row = tweet_lookup[current]
            turns.append({
                "tweet_id": str(row["tweet_id"]),
                "author_id": str(row["author_id"]),
                "text": str(row["text"]),
                "created_at": str(row["created_at"]),
                "inbound": bool(row["inbound"]),
            })
            # Add children in order
            for child_id in children.get(current, []):
                if child_id not in visited:
                    queue.append(child_id)

        if len(turns) == 0:
            continue

        has_amazon = any(t["author_id"] == "AmazonHelp" for t in turns)

        # has_followup: did customer send a message AFTER Amazon's first reply?
        has_followup = False
        seen_amazon = False
        for t in turns:
            if t["author_id"] == "AmazonHelp":
                seen_amazon = True
            elif seen_amazon and t["inbound"]:
                has_followup = True
                break

        threads.append({
            "thread_id": root_id,
            "turns": turns,
            "num_turns": len(turns),
            "has_amazon_reply": has_amazon,
            "has_followup": has_followup,
        })

    return threads


# ── Main pipeline ────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Load
    df = load_csv(args.data_dir)

    # 2. Filter
    df = filter_amazon(df)

    # 3. Reconstruct threads
    threads = reconstruct_threads(df)
    print(f"\n  → {len(threads):,} threads reconstructed total.")

    # 4. Filter: keep only threads with at least one AmazonHelp reply
    threads = [t for t in threads if t["has_amazon_reply"]]
    print(f"  → {len(threads):,} threads with at least one AmazonHelp reply.")

    # 5. Filter: keep only threads with at least 2 turns (need customer msg + reply)
    threads = [t for t in threads if t["num_turns"] >= 2]
    print(f"  → {len(threads):,} threads with ≥2 turns (meaningful conversations).")

    # 6. Sample
    random.seed(args.seed)
    if len(threads) > args.max_threads:
        threads = random.sample(threads, args.max_threads)
        print(f"  → Sampled down to {len(threads):,} threads (seed={args.seed}).")
    else:
        print(f"  → Keeping all {len(threads):,} threads (under max of {args.max_threads}).")

    # 7. Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "amazon_threads.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")
    print(f"\n  ✓ Saved {len(threads):,} threads to {out_path}")

    # 8. Sanity stats
    turns_list = [t["num_turns"] for t in threads]
    followup_pct = sum(1 for t in threads if t["has_followup"]) / len(threads) * 100

    print("\n" + "=" * 60)
    print("SANITY STATS")
    print("=" * 60)
    print(f"  Threads saved          : {len(threads):,}")
    print(f"  Avg turns per thread   : {mean(turns_list):.1f}")
    print(f"  Median turns per thread: {median(turns_list):.1f}")
    print(f"  % with customer follow-up after Amazon reply: {followup_pct:.1f}%")
    print()

    # Turn distribution
    from collections import Counter
    turn_dist = Counter(turns_list)
    print("  Turn count distribution:")
    for k in sorted(turn_dist.keys())[:10]:
        print(f"    {k} turns: {turn_dist[k]:,} threads")
    if len(turn_dist) > 10:
        print(f"    … and {len(turn_dist) - 10} more buckets")
    print()

    # Show 3 example threads
    examples = [threads[0], threads[len(threads) // 2], threads[-1]]
    for i, ex in enumerate(examples):
        first_customer = next(
            (t["text"][:100] for t in ex["turns"] if t["inbound"]),
            "(no inbound turn found)"
        )
        print(f"  Example {i + 1}: thread_id={ex['thread_id']}, "
              f"turns={ex['num_turns']}, followup={ex['has_followup']}")
        print(f"    First customer msg: {first_customer}…")
        print()


if __name__ == "__main__":
    main()
