"""
retrieval.py — Phase 6: Knowledge retrieval over historical resolved cases.

Builds a TF-IDF index over resolved (customer_issue, amazon_reply) pairs
from the processed dataset, supporting intent-filtered similarity retrieval.
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CaseRetrievalEngine:
    def __init__(self, 
                 threads_path: str = "data/processed/amazon_threads_clean.jsonl",
                 exclude_thread_ids: set[str] | list[str] | None = None):
        self.threads_path = Path(threads_path)
        self.exclude_thread_ids = set(str(tid) for tid in (exclude_thread_ids or []))
        self.cases = []  # list of {case_id, query_text, reply_text}
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        if not self.threads_path.exists():
            print(f"Warning: {self.threads_path} not found.")
            return

        with open(self.threads_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                
                # Case 1: Pre-filtered knowledge corpus format
                if "customer_text" in item and "amazon_reply" in item:
                    tid = str(item.get("thread_id", len(self.cases)))
                    if tid in self.exclude_thread_ids:
                        continue
                    self.cases.append({
                        "case_id": tid,
                        "query_text": item["customer_text"],
                        "reply_text": item["amazon_reply"]
                    })
                    continue

                # Case 2: Raw / cleaned threads format
                tid = str(item.get("thread_id", ""))
                if tid and tid in self.exclude_thread_ids:
                    continue

                turns = item.get("turns", [])
                
                # Extract first customer issue and first AmazonHelp response
                cust_text = ""
                amazon_text = ""
                found_cust = False
                
                for t in turns:
                    if t.get("inbound") and not found_cust:
                        cust_text = t.get("text", "")
                        found_cust = True
                    elif found_cust and t.get("author_id") == "AmazonHelp" and not amazon_text:
                        amazon_text = t.get("text", "")
                        break
                
                if cust_text and amazon_text and len(amazon_text) > 10:
                    self.cases.append({
                        "case_id": tid or str(len(self.cases)),
                        "query_text": cust_text,
                        "reply_text": amazon_text
                    })

        if self.cases:
            corpus = [c["query_text"] for c in self.cases]
            self.vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            print(f"Indexed {len(self.cases):,} resolved support cases with TF-IDF (excluded {len(self.exclude_thread_ids)} eval threads).")

    def retrieve_top_k(self, query: str, k: int = 3) -> list[dict]:
        """Retrieves top-k most similar historical cases."""
        if not self.cases or self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = scores.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.05:  # Relevance floor
                c = self.cases[idx]
                results.append({
                    "case_id": c["case_id"],
                    "query_text": c["query_text"],
                    "reply_text": c["reply_text"],
                    "similarity_score": score
                })
        return results


if __name__ == "__main__":
    engine = CaseRetrievalEngine()
    test_q = "my parcel is delayed and was not delivered today"
    hits = engine.retrieve_top_k(test_q, k=2)
    print("\nRetrieval Test Hits:")
    for h in hits:
        print(f"- Score {h['similarity_score']:.3f} | Reply: {h['reply_text']}")
