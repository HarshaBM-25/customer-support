"""
taxonomy.py — Phase 3: Intent taxonomy exploration and definition.

Samples customer-initiated messages from cleaned threads, uses the LLM to
propose and consolidate intent categories, then saves a final taxonomy with
definitions and examples.

Two-step design (can be run independently):
  python -m src.taxonomy --step sample    # extract messages, no LLM needed
  python -m src.taxonomy --step discover  # LLM proposes + consolidates intents
  python -m src.taxonomy                  # runs both steps end-to-end

Usage:
    python -m src.taxonomy
    python -m src.taxonomy --sample-size 400 --step sample
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Discover intent taxonomy from customer messages")
    p.add_argument("--input", type=str,
                   default="./data/processed/amazon_threads_clean.jsonl",
                   help="Cleaned threads JSONL (default: ./data/processed/amazon_threads_clean.jsonl)")
    p.add_argument("--output-dir", type=str, default="./data/processed",
                   help="Where to save taxonomy.json and samples (default: ./data/processed)")
    p.add_argument("--sample-size", type=int, default=400,
                   help="Number of customer messages to sample (default: 400)")
    p.add_argument("--batch-size", type=int, default=25,
                   help="Messages per LLM batch (default: 25)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--step", type=str, choices=["sample", "discover", "all"],
                   default="all", help="Which step to run (default: all)")
    return p.parse_args()


# ── Step 1: Sample customer messages ─────────────────────────────────────────

def sample_customer_messages(input_path: Path, output_dir: Path,
                              sample_size: int, seed: int) -> list[dict]:
    """Extract first inbound customer message from each thread, sample N."""
    if not input_path.exists():
        print(f"ERROR: Input file not found at {input_path}")
        print("Run `python -m src.clean` first.")
        sys.exit(1)

    threads = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))

    print(f"Loaded {len(threads):,} threads")

    # Extract first customer (inbound) message from each thread
    customer_messages = []
    for thread in threads:
        for turn in thread["turns"]:
            if turn["inbound"]:
                customer_messages.append({
                    "thread_id": thread["thread_id"],
                    "tweet_id": turn["tweet_id"],
                    "text": turn["text"],
                    "text_raw": turn.get("text_raw", turn["text"]),
                    "num_turns": thread["num_turns"],
                    "has_followup": thread["has_followup"],
                })
                break  # only first customer message per thread

    print(f"  → {len(customer_messages):,} threads have a customer-initiated first message")

    # Sample
    random.seed(seed)
    if len(customer_messages) > sample_size:
        sample = random.sample(customer_messages, sample_size)
    else:
        sample = customer_messages
    print(f"  → Sampled {len(sample):,} messages (seed={seed})")

    # Save sample
    sample_path = output_dir / "taxonomy_sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for msg in sample:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved sample to {sample_path}")

    return sample


# ── Step 2: LLM-based taxonomy discovery ────────────────────────────────────

BATCH_SYSTEM_PROMPT = """You are an automated API backend that classifies customer support messages into intent categories.
CRITICAL: You must return ONLY a JSON object. No preamble, no chain of thought, no conversation, no markdown fences outside the JSON.
Schema:
{
  "categories_seen": [
    {
      "category": "short_snake_case_name",
      "description": "One-line definition of this intent",
      "count": 3
    }
  ]
}

Rules:
- 3 to 7 high-level intent categories per batch
- snake_case names (e.g. delivery_issue, refund_request, account_access, order_status, product_question, general_complaint, billing_issue)
- Include "other_unclear" for unclassifiable or promotional tweets"""

CONSOLIDATION_SYSTEM_PROMPT = """You are an API backend that consolidates customer support intent categories.
CRITICAL: Return ONLY a JSON object. No preamble, no commentary, no markdown fences outside the JSON.
Schema:
{
  "taxonomy": [
    {
      "intent": "snake_case_name",
      "definition": "Clear one-line definition of when to use this category",
      "merged_from": ["original_category_1", "original_category_2"],
      "examples_description": "Brief description of typical messages"
    }
  ]
}

Rules:
- 6 to 9 distinct, mutually exclusive, actionable support intents
- Must include "other_unclear"
- Use standard snake_case names"""


def discover_taxonomy(sample: list[dict], output_dir: Path,
                       batch_size: int) -> dict:
    """Use LLM to propose and consolidate intent categories."""
    from src.llm_client import call_llm_json

    print("\n" + "=" * 60)
    print("STEP 2a: Batch intent discovery via LLM")
    print("=" * 60)

    all_categories = []  # list of {category, description, count}
    # To be fast, diverse, and robust, use 6 batches of 25 messages (150 messages total)
    discovery_sample = sample[:150]
    batches = [discovery_sample[i:i+batch_size] for i in range(0, len(discovery_sample), batch_size)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="LLM batches")):
        msg_text = "\n".join(
            f"[{i}] {msg['text'][:140]}"
            for i, msg in enumerate(batch)
        )
        prompt = (
            f"Categorize the following {len(batch)} customer support tweets sent to AmazonHelp into 3-7 primary intent categories.\n\n"
            f"Messages:\n{msg_text}\n\n"
            f"Return JSON strictly matching the schema with key 'categories_seen'."
        )

        try:
            result = call_llm_json(
                prompt=prompt,
                system_prompt=BATCH_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=2048,
            )
            cats = result.get("categories_seen", [])
            all_categories.extend(cats)
            print(f"  Batch {batch_idx+1}/{len(batches)}: found {len(cats)} categories")
        except Exception as e:
            print(f"  Batch {batch_idx+1} FAILED: {e}")

    # ── Aggregate discovered categories ──────────────────────────────────
    print(f"\n  → Total raw categories discovered: {len(all_categories)}")

    # Count category name frequencies
    cat_counter = Counter()
    cat_descriptions = {}
    for cat in all_categories:
        name = cat.get("category", "unknown")
        count = cat.get("count", 1)
        cat_counter[name] += count
        if name not in cat_descriptions:
            cat_descriptions[name] = cat.get("description", "")

    print(f"  → Unique category names: {len(cat_counter)}")
    print("\n  Top 20 categories by frequency:")
    for name, count in cat_counter.most_common(20):
        desc = cat_descriptions.get(name, "")
        print(f"    {count:3d}x  {name}: {desc[:80]}")

    # Save raw discoveries
    raw_path = output_dir / "taxonomy_raw_categories.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            "raw_categories": all_categories,
            "category_counts": dict(cat_counter.most_common()),
            "category_descriptions": cat_descriptions,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved raw categories to {raw_path}")

    # ── Consolidation via LLM ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2b: Consolidating into final taxonomy via LLM")
    print("=" * 60)

    consolidation_input = "\n".join(
        f"- {name} ({count}x): {cat_descriptions.get(name, 'no description')}"
        for name, count in cat_counter.most_common()
    )

    prompt = (
        f"Here are the candidate intent categories discovered from {len(sample)} "
        f"AmazonHelp customer messages, with their frequencies:\n\n"
        f"{consolidation_input}\n\n"
        f"Consolidate these into a final taxonomy of 6-10 intents."
    )

    taxonomy_result = call_llm_json(
        prompt=prompt,
        system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=2048,
    )

    taxonomy = taxonomy_result.get("taxonomy", [])
    print(f"\n  → Final taxonomy has {len(taxonomy)} intents:")
    for t in taxonomy:
        print(f"    • {t['intent']}: {t['definition'][:80]}")

    return taxonomy_result


# ── Step 3: Add examples and save final taxonomy ────────────────────────────

def finalize_taxonomy(taxonomy_result: dict, sample: list[dict],
                       output_dir: Path) -> None:
    """Assign example messages to each intent and save final taxonomy.json."""
    from src.llm_client import call_llm_json

    taxonomy = taxonomy_result.get("taxonomy", [])
    intent_names = [t["intent"] for t in taxonomy]

    print("\n" + "=" * 60)
    print("STEP 3: Assigning examples to each intent")
    print("=" * 60)

    # Use LLM to classify a subset of sample messages into the final taxonomy
    # to get 2 good examples per intent
    example_batch = sample[:40]
    msg_text = "\n".join(
        f"[{i}] {msg['text'][:120]}"
        for i, msg in enumerate(example_batch)
    )

    tax_text = "\n".join(f"- {t['intent']}: {t['definition']}" for t in taxonomy)
    prompt = (
        f"Given these {len(example_batch)} messages:\n{msg_text}\n\n"
        f"And this taxonomy:\n{tax_text}\n\n"
        f"Select 1 or 2 representative message index numbers for each intent.\n"
        f"Return ONLY valid JSON matching this schema: {{\"intent_examples\": {{\"{taxonomy[0]['intent']}\": [0, 1]}}}}"
    )

    try:
        examples_result = call_llm_json(
            prompt=prompt,
            system_prompt="You are an automated categorization backend. Return ONLY a JSON object with 'intent_examples'.",
            temperature=0.1,
            max_tokens=2048,
        )
        intent_examples = examples_result.get("intent_examples", {})
    except Exception as e:
        print(f"  Warning: example assignment fallback triggered ({e})")
        intent_examples = {}

    # Build final taxonomy with examples
    final_taxonomy = []
    for t in taxonomy:
        intent = t["intent"]
        definition = t["definition"]
        merged = t.get("merged_from", [])
        examples = []

        indices = intent_examples.get(intent, [])
        for idx in indices[:2]:
            if isinstance(idx, int) and 0 <= idx < len(example_batch):
                examples.append(example_batch[idx]["text"][:200])

        # Pad with placeholder if we don't have 2 examples
        while len(examples) < 2:
            examples.append("(example to be added during labeling)")

        final_taxonomy.append({
            "intent": intent,
            "definition": definition,
            "merged_from": merged,
            "examples": examples,
        })

    # Save
    taxonomy_path = output_dir / "taxonomy.json"
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_intents": len(final_taxonomy),
            "intents": final_taxonomy,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved final taxonomy to {taxonomy_path}")
    print(f"\n  Final taxonomy ({len(final_taxonomy)} intents):")
    for t in final_taxonomy:
        print(f"\n    [{t['intent']}]")
        print(f"      Definition: {t['definition']}")
        print(f"      Merged from: {t['merged_from']}")
        for j, ex in enumerate(t["examples"]):
            print(f"      Example {j+1}: {ex[:100]}…")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    if args.step in ("sample", "all"):
        sample = sample_customer_messages(
            input_path, output_dir, args.sample_size, args.seed
        )
    else:
        # Load previously saved sample
        sample_path = output_dir / "taxonomy_sample.jsonl"
        if not sample_path.exists():
            print(f"ERROR: No sample found at {sample_path}. Run --step sample first.")
            sys.exit(1)
        sample = []
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                sample.append(json.loads(line))
        print(f"Loaded {len(sample):,} previously sampled messages")

    if args.step in ("discover", "all"):
        taxonomy_result = discover_taxonomy(sample, output_dir, args.batch_size)
        finalize_taxonomy(taxonomy_result, sample, output_dir)


if __name__ == "__main__":
    main()
