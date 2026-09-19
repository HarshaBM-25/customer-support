"""
reply_agent.py — Phase 6: Core Customer Support Agent Pipeline.

Full pipeline per customer message:
  1. Intent classification via LLM (classifier.py)
  2. Grounded knowledge retrieval over resolved cases (retrieval.py)
  3. LLM reply drafting grounded in retrieved support examples
  4. Explicit multi-tier routing (hard rules for safety/legal/security + confidence thresholds)
  5. Full structured audit trace logging to data/results/agent_output.jsonl
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.classifier import classify_intent
from src.retrieval import CaseRetrievalEngine
from src.llm_client import call_llm_json

DRAFT_SYSTEM_PROMPT = """You are an official Amazon Customer Support assistant on Twitter/X (@AmazonHelp).
Your task is to write a helpful, professional, and empathetic reply to the customer's inquiry.

Grounding rules:
- Ground your response in the historical resolved support cases provided below if relevant.
- Keep the response concise, clear, and action-oriented (appropriate for social media support).
- If specific account details or orders need private lookup, instruct the customer to reach out via Direct Message (DM) or check 'Your Orders'.
- Never make false promises or fabricate tracking details.

Respond in JSON format:
{
  "reply": "<drafted support reply>",
  "tone": "<empathetic|professional|direct>",
  "action_suggested": "<brief summary of next step provided to customer>"
}
"""

HARD_ESCALATE_KEYWORDS = [
    "sue", "lawsuit", "legal action", "court", "lawyer", "attorney", "police", "fir",
    "injury", "injured", "safety hazard", "fire", "explosion", "electric shock",
    "hacked", "account stolen", "unauthorized transaction", "identity theft",
    "fraud", "scam", "consumer court"
]


class AmazonSupportAgent:
    def __init__(self, 
                 retrieval_corpus_path: str = "data/processed/knowledge_corpus.jsonl",
                 exclude_thread_ids: set[str] | list[str] | None = None):
        # Prefer pre-filtered knowledge_corpus if available, otherwise filter on the fly
        corpus_file = Path(retrieval_corpus_path)
        if not corpus_file.exists():
            corpus_file = Path("data/processed/amazon_threads_clean.jsonl")
        self.retrieval_engine = CaseRetrievalEngine(
            threads_path=str(corpus_file),
            exclude_thread_ids=exclude_thread_ids
        )

    def process_message(self, text: str) -> dict:
        text_lower = text.lower()

        # Step 1: Pre-LLM Hard Rule Checks (Immediate Escalation with word boundary)
        import re
        for kw in HARD_ESCALATE_KEYWORDS:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text_lower):
                return {
                    "intent": "account_issue" if "hack" in kw or "stolen" in kw else "other_unclear",
                    "confidence": 1.0,
                    "retrieved_cases": [],
                    "draft_reply": "We take this matter very seriously. A specialist from our senior support team has been assigned to assist you immediately via Direct Message.",
                    "route": "ESCALATE",
                    "route_reason": f"Hard safety/legal/security trigger matched: '{kw}'"
                }

        # Step 2: Intent Classification
        cls_result = classify_intent(text)
        intent = cls_result["intent"]
        confidence = cls_result["confidence"]

        # Step 3: Retrieval of Grounding Cases
        retrieved_cases = self.retrieval_engine.retrieve_top_k(text, k=2)

        # Step 4: Draft Reply via LLM
        grounding_context = "\n".join([
            f"- Historical Issue: {c['query_text']}\n  Amazon Resolution: {c['reply_text']}"
            for c in retrieved_cases
        ]) if retrieved_cases else "No direct historical match found. Use standard Amazon customer support best practices."

        user_prompt = f"""Customer message:
\"\"\"{text}\"\"\"

Classified Intent: {intent} (Confidence: {confidence:.2f})

Historical Grounding Examples:
{grounding_context}

Draft the best support response."""

        try:
            draft_res = call_llm_json(
                prompt=user_prompt,
                system_prompt=DRAFT_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=600
            )
            draft_reply = draft_res.get("reply", "Hello, please DM us your order details so we can assist you directly.")
        except Exception as e:
            draft_reply = "Hello, please reach out to us via Direct Message with your order ID so our team can look into this for you."

        # Step 5: Post-classification Routing Decision
        if intent == "other_unclear":
            route = "ESCALATE"
            route_reason = "Intent is unclear or ambiguous; escalated for human triage."
        elif intent == "account_issue":
            route = "ESCALATE"
            route_reason = "Account security/access issues require human agent authentication."
        elif confidence < 0.60:
            route = "ESCALATE"
            route_reason = f"Classification confidence ({confidence:.2f}) below operational threshold (0.60)."
        else:
            route = "AUTO"
            route_reason = f"High-confidence automated resolution available for intent '{intent}'."

        return {
            "intent": intent,
            "confidence": confidence,
            "retrieved_cases": [c["case_id"] for c in retrieved_cases],
            "draft_reply": draft_reply,
            "route": route,
            "route_reason": route_reason
        }


def run_agent_evaluation(golden_set_path: str = "data/eval/golden_set_human.jsonl",
                          output_path: str = "data/results/agent_eval.jsonl"):
    golden_file = Path(golden_set_path)
    if not golden_file.exists():
        # Fallback to legacy path if human golden set is not yet generated
        golden_file = Path("data/eval/golden_set.jsonl")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(golden_file, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    # Explicitly extract golden set thread IDs to enforce strict zero-leakage retrieval
    eval_tids = set(str(ex.get("thread_id", "")) for ex in examples if ex.get("thread_id"))
    agent = AmazonSupportAgent(exclude_thread_ids=eval_tids)

    # Load existing processed IDs if resuming
    processed = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    processed[rec["id"]] = rec
        if processed:
            print(f"Resuming evaluation: {len(processed)}/{len(examples)} examples already completed.")

    print(f"\nProcessing {len(examples)} golden set examples through Core Agent Pipeline (leak-free)...")

    with open(out_file, "a", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="Agent Processing"):
            eid = ex["id"]
            if eid in processed:
                continue

            res = agent.process_message(ex["cleaned_text"])
            
            # Ground truth is human label if present, otherwise fallback
            human_intent = ex.get("human_intent") or ex.get("true_intent", "")
            human_route = ex.get("human_route") or ex.get("correct_route", "")
            human_reason = ex.get("human_route_reason") or ex.get("route_reason", "")

            out_record = {
                "id": eid,
                "thread_id": ex["thread_id"],
                "cleaned_text": ex["cleaned_text"],
                "human_intent": human_intent,
                "human_route": human_route,
                "human_route_reason": human_reason,
                "pred_intent": res["intent"],
                "confidence": res["confidence"],
                "retrieved_cases": res["retrieved_cases"],
                "pred_reply": res["draft_reply"],
                "pred_route": res["route"],
                "route_reason": res["route_reason"]
            }
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\n✓ Core Agent pipeline completed. Output saved to {out_file}")


def interactive_mode():
    agent = AmazonSupportAgent()
    print("\n" + "=" * 60)
    print("🤖 AmazonHelp AI Support Agent — Interactive CLI Preview")
    print("=" * 60)
    print("Type a customer message to preview agent classification, grounding RAG, reply, and routing.")
    print("Type 'exit' or 'quit' to exit.\n")

    while True:
        try:
            query = input("\n👤 Customer: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive preview.")
                break

            print("\n🔄 Processing inquiry...")
            res = agent.process_message(query)

            print("-" * 60)
            print(f"📌 Classified Intent : {res['intent']} (Confidence: {res['confidence']:.2f})")
            print(f"🚦 Routing Decision  : {res['route']} ({res['route_reason']})")
            print(f"📚 Grounding Case IDs: {res['retrieved_cases']}")
            print(f"💬 Drafted Response  :\n{res['draft_reply']}")
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive preview.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Core Agent Pipeline")
    parser.add_argument("--golden-set", type=str, default="data/eval/golden_set_human.jsonl")
    parser.add_argument("--output", type=str, default="data/results/agent_eval.jsonl")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI demo mode")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_agent_evaluation(args.golden_set, args.output)
