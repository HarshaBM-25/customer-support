"""
classifier.py — Phase 6: LLM-based Intent Classifier with Confidence Scoring.

Classifies incoming customer messages into one of the 8 taxonomy intents
using NVIDIA Nemotron with structured JSON output and confidence score.
"""

import json
from pathlib import Path
from src.llm_client import call_llm_json

TAXONOMY_PATH = Path("data/processed/taxonomy.json")

def load_taxonomy():
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Taxonomy file not found at {TAXONOMY_PATH}")
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

TAXONOMY_DATA = load_taxonomy()
VALID_INTENTS = [t["intent"] for t in TAXONOMY_DATA["intents"]]
INTENT_DEFINITIONS = "\n".join([f"- {t['intent']}: {t['definition']}" for t in TAXONOMY_DATA["intents"]])

SYSTEM_PROMPT = f"""You are an expert customer intent classifier for Amazon customer support.
Classify the customer's message into exactly one of the following 8 intents:

{INTENT_DEFINITIONS}

Respond strictly in valid JSON with schema:
{{
  "intent": "<intent_name>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<brief justification>"
}}
"""

def classify_intent(text: str) -> dict:
    """Classifies a customer message into an intent with confidence score."""
    prompt = f"Customer message:\n\"\"\"{text}\"\"\"\n\nClassify the primary intent."
    try:
        res = call_llm_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=500
        )
        intent = res.get("intent", "other_unclear")
        if intent not in VALID_INTENTS:
            # Fallback if model hallucinated a non-standard name
            matched = [v for v in VALID_INTENTS if v in intent]
            intent = matched[0] if matched else "other_unclear"
        
        confidence = float(res.get("confidence", 0.8))
        reason = res.get("reason", "Classified by LLM")
        return {
            "intent": intent,
            "confidence": min(max(confidence, 0.0), 1.0),
            "reason": reason
        }
    except Exception as e:
        return {
            "intent": "other_unclear",
            "confidence": 0.0,
            "reason": f"Classification error: {e}"
        }

if __name__ == "__main__":
    test_msg = "My package has not arrived and tracking is stuck for 3 days!"
    print(f"Test Classification: {classify_intent(test_msg)}")
