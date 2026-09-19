"""
baselines.py — Phase 5: Implementation of Trivial and Simple baseline models.

Baselines implemented:
1. Trivial Baseline:
   - Intent: Predicts majority intent (or fixed default).
   - Reply: Fixed generic canned message.
   - Routing: Always ESCALATE (safest trivial dumb strategy).

2. Simple Baseline:
   - Intent: Keyword/heuristic matching.
   - Reply: Intent-specific template reply.
   - Routing: Rule-based keyword escalation (legal, security, dispute keywords).

Usage:
    python -m src.baselines
"""

import argparse
import json
from collections import Counter
from pathlib import Path


# ── Keyword & Template Definitions ──────────────────────────────────────────

INTENT_KEYWORDS = {
    "return_refund": ["refund", "return", "replace", "damaged", "broken", "defective", "money back", "faulty"],
    "shipping_delivery": ["delivered", "delivery", "late", "driver", "courier", "package", "dispatch", "bluedart", "tnt", "ups", "fedex", "shipment"],
    "order_status": ["track", "where is", "status", "order id", "confirmation", "expected", "when will", "dispatch status"],
    "payment_issue": ["charge", "charged", "billing", "card", "bank", "cashback", "pay balance", "gst", "deducted", "payment", "unauthorized"],
    "account_issue": ["login", "password", "hacked", "account", "otp", "email changed", "sign in", "blocked", "access"],
    "technical_support": ["app", "crash", "website", "cart", "server", "kindle error", "fire tv", "alexa", "echo", "error", "bug"],
    "product_question": ["price", "offer", "discount", "warranty", "available", "specs", "feature", "catalogue", "prime video", "compatibility"],
}

ESCALATE_KEYWORDS = [
    "sue", "legal", "court", "lawyer", "police", "fraud", "cheat", "scam", "harass",
    "hacked", "stolen", "unauthorized", "dispute", "complaint", "worst", "threat"
]

CANNED_REPLIES = {
    "order_status": "Hello! Please share your order number via Direct Message so we can look into the status for you.",
    "shipping_delivery": "We apologize for the delivery delay. Please send us your tracking and order details via DM so we can assist.",
    "return_refund": "We're sorry for the inconvenience. You can request a return/refund under 'Your Orders' > 'Problem with Order' or DM us.",
    "payment_issue": "We apologize for the billing issue. Please check your bank transaction and DM us your order ID securely.",
    "product_question": "Thanks for reaching out! You can view detailed product specifications and availability on our product page.",
    "account_issue": "For your security, please verify your credentials or visit our secure account recovery portal.",
    "technical_support": "We apologize for the technical issue. Please try restarting your app/device or clearing your cache.",
    "other_unclear": "Hello! Could you please provide more details regarding your query so we can assist you better?"
}

GENERIC_TRIVIAL_REPLY = "Hello, thank you for reaching out. Please send us a Direct Message with your order details so we can assist."


# ── Baseline Logic ──────────────────────────────────────────────────────────

def run_trivial_baseline(examples: list[dict], majority_intent: str) -> list[dict]:
    """Trivial Baseline: always majority intent, generic template, always ESCALATE."""
    results = []
    for ex in examples:
        human_intent = ex.get("human_intent") or ex.get("true_intent", "")
        human_route = ex.get("human_route") or ex.get("correct_route", "")
        results.append({
            "id": ex["id"],
            "thread_id": ex["thread_id"],
            "cleaned_text": ex["cleaned_text"],
            "human_intent": human_intent,
            "human_route": human_route,
            "pred_intent": majority_intent,
            "pred_reply": GENERIC_TRIVIAL_REPLY,
            "pred_route": "ESCALATE",
            "route_reason": "Trivial baseline default: always escalate all inquiries to human agents"
        })
    return results


def run_simple_baseline(examples: list[dict]) -> list[dict]:
    """Simple Baseline: keyword-based intent, per-intent canned template, rule-based escalation."""
    results = []
    for ex in examples:
        text = ex["cleaned_text"].lower()

        # 1. Intent classification via keywords
        matched_intent = "other_unclear"
        best_match_count = 0
        for intent, kws in INTENT_KEYWORDS.items():
            count = sum(1 for kw in kws if kw in text)
            if count > best_match_count:
                best_match_count = count
                matched_intent = intent

        # 2. Reply selection
        pred_reply = CANNED_REPLIES.get(matched_intent, CANNED_REPLIES["other_unclear"])

        # 3. Rule-based routing
        if matched_intent in ["account_issue", "other_unclear"] or any(w in text for w in ESCALATE_KEYWORDS):
            pred_route = "ESCALATE"
            route_reason = f"Keyword/rule triggered escalation (intent: {matched_intent})"
        else:
            pred_route = "AUTO"
            route_reason = f"Standard pattern matched ({matched_intent}), auto-reply assigned"

        human_intent = ex.get("human_intent") or ex.get("true_intent", "")
        human_route = ex.get("human_route") or ex.get("correct_route", "")

        results.append({
            "id": ex["id"],
            "thread_id": ex["thread_id"],
            "cleaned_text": ex["cleaned_text"],
            "human_intent": human_intent,
            "human_route": human_route,
            "pred_intent": matched_intent,
            "pred_reply": pred_reply,
            "pred_route": pred_route,
            "route_reason": route_reason
        })
    return results


# ── Execution ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baseline models on golden eval set")
    parser.add_argument("--golden-set", type=str, default="data/eval/golden_set_human.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/results")
    args = parser.parse_args()

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        golden_path = Path("data/eval/golden_set.jsonl")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not golden_path.exists():
        print(f"Error: Golden eval set not found at {golden_path}")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples from {golden_path}")

    # Determine majority class in evaluation set based on human labels
    intent_counts = Counter(ex.get("human_intent") or ex.get("true_intent", "shipping_delivery") for ex in examples)
    majority_intent = intent_counts.most_common(1)[0][0]

    # Run Trivial Baseline
    trivial_results = run_trivial_baseline(examples, majority_intent)
    trivial_out = out_dir / "baseline_trivial.jsonl"
    with open(trivial_out, "w", encoding="utf-8") as f:
        for r in trivial_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ Saved Trivial Baseline predictions ({len(trivial_results)}) to {trivial_out}")

    # Run Simple Baseline
    simple_results = run_simple_baseline(examples)
    simple_out = out_dir / "baseline_simple.jsonl"
    with open(simple_out, "w", encoding="utf-8") as f:
        for r in simple_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ Saved Simple Baseline predictions ({len(simple_results)}) to {simple_out}")

    # Quick Summary
    print("\n" + "=" * 50)
    print("BASELINE RUN SUMMARY")
    print("=" * 50)
    # Trivial accuracy
    t_intent_acc = sum(1 for r in trivial_results if r["pred_intent"] == r["human_intent"]) / len(trivial_results)
    t_route_acc = sum(1 for r in trivial_results if r["pred_route"] == r["human_route"]) / len(trivial_results)
    print(f"Trivial Baseline -> Intent Acc: {t_intent_acc*100:.1f}%, Route Acc: {t_route_acc*100:.1f}%")

    # Simple accuracy
    s_intent_acc = sum(1 for r in simple_results if r["pred_intent"] == r["human_intent"]) / len(simple_results)
    s_route_acc = sum(1 for r in simple_results if r["pred_route"] == r["human_route"]) / len(simple_results)
    print(f"Simple Baseline  -> Intent Acc: {s_intent_acc*100:.1f}%, Route Acc: {s_route_acc*100:.1f}%")


if __name__ == "__main__":
    main()
