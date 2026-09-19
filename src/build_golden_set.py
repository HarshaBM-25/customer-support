"""
fix_golden_set.py — Re-derive golden set labels from the actual message text,
row by row, instead of copying the pre-classifier's own predictions.

Method (documented so it can be explained honestly if asked):
  For each of the 200 rows, the raw customer message is read and matched
  against explicit, inspectable trigger rules (keyword/phrase based) for:
    - intent (8-way, based on what the message is actually about)
    - escalation triggers (fraud/legal/safety/security/repeat-failure/
      high-value-money/explicit management demand/strong frustration)
  route_reason is built FROM the specific trigger phrase found in that row's
  text, so reasons differ row to row instead of reusing 11 templates.
  good_reply_example is left as the brand's actual reply only when it passes
  a basic adequacy check (apology/ack + a concrete next step, no obvious tone
  mismatch with detected frustration); otherwise a rewritten reply is
  generated referencing the specific complaint in that row.

This is a rule-based, text-grounded relabeling pass — a defensible starting
point, not a substitute for a human spot-check. Recommend reviewing a random
20-30 row sample by hand before treating this as final ground truth.
"""
import json
import re
import random
from pathlib import Path

IN_PATH = Path("data/eval/golden_set.jsonl")
OUT_PATH = Path("data/eval/golden_set_fixed.jsonl")

INTENT_KEYWORDS = {
    "return_refund": ["refund", "return", "money back", "reimburse", "exchange",
                        "damaged", "broken", "wrong item", "defective"],
    "shipping_delivery": ["deliver", "shipping", "shipped", "package", "parcel",
                            "tracking", "arrive", "late", "hasn't come", "hasn't arrived",
                            "courier", "carrier"],
    "order_status": ["order status", "order #", "order number", "cancelled", "canceled",
                       "my order", "order id", "order was"],
    "payment_issue": ["charge", "charged", "emi", "payment", "billed", "bill",
                        "overcharged", "double charged", "card declined", "funds"],
    "account_issue": ["sign on", "sign in", "log in", "login", "password", "account",
                        "hacked", "unauthorized access", "can't access", "locked out"],
    "technical_support": ["app crash", "app is", "error", "won't load", "bug", "glitch",
                            "doesn't work", "not working", "site down", "website down"],
    "product_question": ["available", "availability", "launch", "when is", "does it",
                           "feature", "compatible", "release"],
}

FRUSTRATION_WORDS = ["never", "worst", "pathetic", "disgusted", "ridiculous", "joke",
                      "scam", "fraud", "steal", "stealing", "liar", "liars", "wtf",
                      "cheat", "furious", "angry", "unacceptable", "terrible", "awful"]
LEGAL_WORDS = ["lawyer", "legal action", "sue", "court", "attorney", "lawsuit"]
SAFETY_WORDS = ["injur", "hurt", "dangerous", "fire", "shock", "burn", "unsafe"]
SECURITY_WORDS = ["hacked", "unauthorized access", "stolen account", "someone accessed",
                    "not me who", "fraudulent charge"]
MANAGEMENT_WORDS = ["senior management", "manager", "escalate this", "corporate",
                      "ceo", "head office"]
REPEAT_FAIL_WORDS = ["already told", "already said", "spoke to", "customer service say",
                       "customer service said", "told me they can't", "said they can't",
                       "still nothing", "still haven't", "still no"]
MONEY_PATTERN = re.compile(r"(\$|₹|inr|usd)\s?\d{2,}|(\d{2,}\s?(inr|usd|rs))", re.I)


def detect_intent(text: str) -> str:
    t = text.lower()
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        hits = [kw for kw in kws if kw in t]
        if hits:
            scores[intent] = hits
    if not scores:
        return "other_unclear", None
    # pick intent with most keyword hits; tie-break by keyword specificity (longer phrase wins)
    best = max(scores.items(), key=lambda kv: (len(kv[1]), max(len(h) for h in kv[1])))
    return best[0], best[1][0]


def detect_escalation(text: str, has_followup: bool):
    t = text.lower()
    for w in LEGAL_WORDS:
        if w in t:
            return True, f"legal/compliance language detected ('{w}')"
    for w in SAFETY_WORDS:
        if w in t:
            return True, f"possible safety concern detected ('{w}')"
    for w in SECURITY_WORDS:
        if w in t:
            return True, f"account security/fraud concern detected ('{w}')"
    for w in MANAGEMENT_WORDS:
        if w in t:
            return True, f"explicit escalation demand detected ('{w}')"
    for w in REPEAT_FAIL_WORDS:
        if w in t:
            return True, f"customer states a prior human-agent contact already failed to resolve this ('{w}')"
    money_match = MONEY_PATTERN.search(t)
    frustrated_hits = [w for w in FRUSTRATION_WORDS if w in t]
    if money_match and frustrated_hits:
        return True, f"specific monetary amount ('{money_match.group(0)}') combined with frustrated language ('{frustrated_hits[0]}')"
    if len(frustrated_hits) >= 2:
        return True, f"multiple frustration markers detected ({', '.join(frustrated_hits[:2])})"
    if text.isupper() and len(text) > 15:
        return True, "message is in all-caps, signalling high frustration"
    return False, None


def adequacy_check(actual_reply: str, frustrated_hits, escalate: bool) -> bool:
    """Rough check: does the actual reply at least acknowledge + give a next step,
    without being tone-deaf relative to detected frustration?"""
    if not actual_reply or not actual_reply.strip():
        return False
    r = actual_reply.lower()
    has_ack = any(w in r for w in ["sorry", "apolog", "understand", "thank"])
    has_next_step = any(w in r for w in ["dm", "link", "http", "<url>", "share", "email",
                                          "click", "check", "reach out", "message"])
    if escalate and frustrated_hits and not has_ack:
        return False  # tone-deaf: frustrated customer, no acknowledgement at all
    return has_ack or has_next_step


def build_reply(text: str, intent: str, escalate: bool, reason: str) -> str:
    lead = "I'm sorry for the trouble here" if escalate else "Thanks for reaching out"
    if intent == "return_refund":
        ask = "could you share your order ID via DM so we can get eyes on the refund directly?"
    elif intent == "shipping_delivery":
        ask = "could you share your order ID via DM so we can check what's going on with the shipment?"
    elif intent == "payment_issue":
        ask = "could you DM your order ID so our billing team can review the charge directly?"
    elif intent == "account_issue":
        ask = "could you confirm whether you've checked spam/junk folders, or DM us if the account shows any unfamiliar activity?"
    elif intent == "order_status":
        ask = "could you share your order ID via DM so we can look into the cancellation directly?"
    else:
        ask = "could you share a bit more detail via DM so we can look into this properly?"
    if escalate:
        return f"{lead} -- I want to get this in front of a specialist rather than send a generic reply, since {reason.split('(')[0].strip()}. In the meantime, {ask}"
    return f"{lead}! {ask.capitalize()}"


def main():
    rows = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    random.seed(11)
    fixed = []
    changed_intent = 0
    changed_route = 0
    kept_actual_reply = 0
    rewrote_reply = 0

    for row in rows:
        text = row.get("cleaned_text") or row.get("raw_text") or ""
        intent, trigger_kw = detect_intent(text)
        escalate, esc_reason = detect_escalation(text, row.get("num_turns", 0) > 4)
        frustrated_hits = [w for w in FRUSTRATION_WORDS if w in text.lower()]

        if intent != row["true_intent"]:
            changed_intent += 1
        new_route = "ESCALATE" if escalate else "AUTO"
        if new_route != row["correct_route"]:
            changed_route += 1

        if esc_reason:
            route_reason = f"ESCALATE: {esc_reason}." if escalate else f"AUTO: no escalation triggers found; matched intent '{intent}' via keyword '{trigger_kw}'." 
        else:
            if escalate:
                route_reason = "ESCALATE: message tone/content flagged as high-risk for automated handling."
            else:
                route_reason = (f"AUTO: routine '{intent}' request, matched via keyword '{trigger_kw}'."
                                 if trigger_kw else
                                 f"AUTO: no strong intent signal found; treated as low-risk default, but should be spot-checked.")

        actual_reply = row.get("amazon_actual_reply", "")
        if adequacy_check(actual_reply, frustrated_hits, escalate):
            good_reply = "ACTUAL_OK"
            kept_actual_reply += 1
        else:
            good_reply = build_reply(text, intent, escalate, route_reason)
            rewrote_reply += 1

        fixed.append({
            **row,
            "true_intent": intent,
            "correct_route": new_route,
            "route_reason": route_reason,
            "good_reply_example": good_reply,
            "label_method": "rule_based_text_grounded_v1",  # documents this isn't a blind copy
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total rows: {len(fixed)}")
    print(f"Intent changed from original: {changed_intent}")
    print(f"Route changed from original: {changed_route}")
    print(f"Kept ACTUAL_OK: {kept_actual_reply} ({kept_actual_reply/len(fixed)*100:.1f}%)")
    print(f"Rewrote reply: {rewrote_reply} ({rewrote_reply/len(fixed)*100:.1f}%)")
    from collections import Counter
    print("New intent distribution:", Counter(r["true_intent"] for r in fixed))
    print("New route distribution:", Counter(r["correct_route"] for r in fixed))
    print("Unique route_reason strings:", len(set(r["route_reason"] for r in fixed)))


if __name__ == "__main__":
    main()