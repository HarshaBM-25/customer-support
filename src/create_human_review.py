import json
import csv
import re
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/eval/golden_set.jsonl")
THREADS_FILE = Path("data/processed/amazon_threads_clean.jsonl")
HUMAN_JSONL = Path("data/eval/golden_set_human.jsonl")
REVIEW_CSV = Path("data/eval/golden_set_human_review.csv")


# ============================================================
# INTENT KEYWORDS
# Used ONLY to generate suggestions.
# These are NOT the final ground-truth labels.
# ============================================================

INTENT_KEYWORDS = {
    "order_status": [
        "where is my order",
        "where's my order",
        "order status",
        "track my order",
        "tracking",
        "order hasn't arrived",
        "order hasnt arrived",
        "package hasn't arrived",
        "package hasnt arrived",
        "delivery status",
    ],

    "shipping_delivery": [
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "courier",
        "late delivery",
        "delivery date",
        "delivery time",
        "arriving",
    ],

    "return_refund": [
        "refund",
        "return",
        "returned",
        "returning",
        "money back",
        "refund status",
        "refund hasn't",
        "refund hasnt",
        "replacement",
        "exchange",
    ],

    "payment_issue": [
        "charged",
        "charge",
        "payment",
        "billing",
        "bill",
        "credit card",
        "debit card",
        "charged twice",
        "double charged",
        "wrong charge",
        "unknown charge",
        "unauthorized charge",
    ],

    "product_question": [
        "product",
        "item",
        "available",
        "availability",
        "price",
        "cost",
        "size",
        "colour",
        "color",
        "model",
        "feature",
        "specification",
        "specs",
    ],

    "account_issue": [
        "account",
        "login",
        "log in",
        "sign in",
        "password",
        "username",
        "locked",
        "account locked",
        "membership",
        "prime membership",
    ],

    "technical_support": [
        "app",
        "website",
        "error",
        "bug",
        "not working",
        "doesn't work",
        "doesnt work",
        "crash",
        "broken",
        "technical",
        "loading",
        "login error",
    ],
}


# ============================================================
# ESCALATION SIGNALS
# ============================================================

ESCALATION_PATTERNS = {
    "frustration": [
        "angry",
        "ridiculous",
        "terrible",
        "worst",
        "unacceptable",
        "disappointed",
        "frustrated",
        "frustrating",
        "fed up",
        "sick of",
    ],

    "legal": [
        "lawyer",
        "attorney",
        "legal action",
        "lawsuit",
        "court",
        "sue",
        "consumer court",
    ],

    "safety": [
        "danger",
        "dangerous",
        "unsafe",
        "injured",
        "injury",
        "fire",
        "hazard",
    ],

    "security": [
        "hacked",
        "hack",
        "stolen account",
        "someone accessed",
        "unauthorized",
        "fraud",
        "scam",
        "security",
    ],

    "management": [
        "manager",
        "supervisor",
        "speak to a manager",
        "speak to supervisor",
        "escalate",
    ],

    "repeat_failure": [
        "again",
        "still not",
        "already contacted",
        "already contacted you",
        "third time",
        "second time",
        "multiple times",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_intent(text):
    """
    Generate a suggested intent.

    IMPORTANT:
    This is only a pre-label suggestion.
    Human reviewer decides the final intent.
    """

    text = normalize(text)

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "other_unclear"

    return best_intent


def detect_escalation(text):
    """
    Generate a suggested routing decision.
    """

    text = normalize(text)

    matched_reasons = []

    for reason, patterns in ESCALATION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                matched_reasons.append(reason)
                break

    # Strong financial/security/safety/legal signals
    if any(
        word in text
        for word in [
            "unauthorized charge",
            "fraud",
            "hacked",
            "stolen account",
            "legal action",
            "lawsuit",
            "dangerous",
            "injured",
        ]
    ):
        return (
            "ESCALATE",
            "Strong safety, security, legal, or financial-risk signal."
        )

    if matched_reasons:
        return (
            "ESCALATE",
            "Possible escalation signal: "
            + ", ".join(matched_reasons)
            + "."
        )

    # Account problems are safer to escalate in this project.
    if "account" in text and any(
        word in text
        for word in [
            "locked",
            "hacked",
            "stolen",
            "can't login",
            "cannot login",
        ]
    ):
        return (
            "ESCALATE",
            "Account/security issue may require human handling."
        )

    return (
        "AUTO",
        "No strong escalation signal detected by the pre-labeler."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}.\n"
            f"Run this script from the project ROOT directory."
        )

    # ----------------------------------------------------------
    # Load threads to build conversation context
    # ----------------------------------------------------------
    threads_lookup = {}
    if THREADS_FILE.exists():
        print(f"Loading threads from {THREADS_FILE}...")
        with THREADS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                thread = json.loads(line)
                tid = str(thread.get("thread_id", ""))
                threads_lookup[tid] = thread
        print(f"  Loaded {len(threads_lookup):,} threads.")
    else:
        print(f"WARNING: {THREADS_FILE} not found. "
              f"conversation_context will be empty.")

    # ----------------------------------------------------------
    # Load golden set examples
    # ----------------------------------------------------------
    rows = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} golden set examples.")

    if len(rows) != 200:
        print(
            f"WARNING: Expected 200 examples, "
            f"but found {len(rows)}."
        )

    # ----------------------------------------------------------
    # Build human review records
    # ----------------------------------------------------------
    # Fields that are old heuristic "ground truth" and should
    # NOT appear in the human review file (they could bias the
    # reviewer).
    OLD_TRUTH_FIELDS = {
        "true_intent", "correct_route", "route_reason",
        "good_reply_example", "predicted_intent",
        "difficulty_tags", "conversation_context",
    }

    human_rows = []

    for row in rows:

        # Prefer cleaned text, otherwise raw text.
        text = row.get("cleaned_text") or row.get("raw_text") or ""

        suggested_intent = detect_intent(text)

        suggested_route, suggested_reason = detect_escalation(text)

        # Build conversation context from thread turns
        tid = str(row.get("thread_id", ""))
        thread = threads_lookup.get(tid, {})
        turns = thread.get("turns", [])
        context_parts = []
        for turn in turns[:6]:
            speaker = "CUSTOMER" if turn.get("inbound") else "AMAZON"
            turn_text = turn.get("text", "")[:300]
            context_parts.append(f"[{speaker}] {turn_text}")
        conversation_context = "\n".join(context_parts)

        # Find Amazon's actual reply (first Amazon turn after
        # the first customer turn)
        amazon_reply = row.get("amazon_actual_reply", "")
        if not amazon_reply:
            found_customer = False
            for turn in turns:
                if turn.get("inbound") and not found_customer:
                    found_customer = True
                    continue
                if found_customer and not turn.get("inbound"):
                    amazon_reply = turn.get("text", "")
                    break

        # Build clean record — exclude old heuristic labels
        new_row = {
            "id": row.get("id"),
            "thread_id": tid,
            "raw_text": row.get("raw_text", ""),
            "cleaned_text": text,
            "conversation_context": conversation_context,
            "amazon_actual_reply": amazon_reply,
            "num_turns": row.get("num_turns",
                                 thread.get("num_turns", 0)),

            # Pre-labeler suggestions (NOT ground truth)
            "suggested_intent": suggested_intent,
            "suggested_route": suggested_route,
            "suggested_route_reason": suggested_reason,

            # HUMAN GROUND TRUTH — to be filled by reviewer
            "human_intent": "",
            "human_route": "",
            "human_route_reason": "",
            "human_reviewed": False,
        }

        human_rows.append(new_row)

    # ========================================================
    # WRITE JSONL
    # ========================================================

    HUMAN_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with HUMAN_JSONL.open("w", encoding="utf-8") as f:

        for row in human_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"\nCreated:")
    print(f"  {HUMAN_JSONL}")

    # ========================================================
    # WRITE CSV FOR EASY HUMAN REVIEW
    # ========================================================

    csv_fields = [
        "id",
        "thread_id",
        "raw_text",
        "cleaned_text",
        "conversation_context",
        "amazon_actual_reply",

        "suggested_intent",
        "suggested_route",
        "suggested_route_reason",

        "human_intent",
        "human_route",
        "human_route_reason",
        "human_reviewed",
    ]

    with REVIEW_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in human_rows:
            writer.writerow(row)

    print(f"  {REVIEW_CSV}")

    print("\n" + "=" * 60)
    print("HUMAN REVIEW FILE CREATED")
    print("=" * 60)

    print("\nYour workflow is now:")

    print("""
1. Open:
   data/eval/golden_set_human_review.csv

2. Read each customer message/conversation.

3. Check the suggested labels.

4. Decide the FINAL human label.

5. Fill:
   human_intent
   human_route
   human_route_reason

6. Set:
   human_reviewed = true

7. Do NOT change:
   suggested_intent
   suggested_route

Those are only pre-labeler suggestions.
""")

    print("Valid intents:")
    for intent in INTENT_KEYWORDS:
        print(f"  - {intent}")

    print("  - other_unclear")

    print("\nValid routes:")
    print("  - AUTO")
    print("  - ESCALATE")


if __name__ == "__main__":
    main()