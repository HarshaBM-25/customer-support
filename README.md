# Automated Customer Support AI Agent for AmazonHelp
> **Hiver SDE Intern Take-Home Project**  
> An intelligent customer service agent for `@AmazonHelp` that combines intent classification, historical knowledge retrieval (RAG), and a multi-tier safety routing policy to automatically resolve support inquiries or safely escalate them to human specialists.

---

## 📌 Project Overview

Customer support on public social channels like Twitter/X requires high accuracy, rapid response times, and strict safety guardrails. Providing an automated wrong answer on a sensitive issue (e.g., an account compromise or legal threat) is significantly worse than escalating the issue to a human agent.

This project implements an end-to-end, production-grade support system that:
1. **Classifies Customer Intent** into an 8-class data-derived taxonomy using NVIDIA Nemotron LLM.
2. **Grounds Responses using RAG** by retrieving historically resolved cases via TF-IDF from a clean corpus of 4,787 customer conversations (with strict zero-leakage separation from evaluation cases).
3. **Drafts Empathetic, Policy-Compliant Responses** that guide customers to secure resolution channels (e.g., Direct Message, 'Your Orders').
4. **Applies Multi-Tier Safety Routing** to automatically approve responses (`AUTO`) or route high-risk/ambiguous inquiries to human specialists (`ESCALATE`).
5. **Evaluates Performance Honestly** against two baselines (Trivial & Simple Rule-based) and a hand-labeled golden evaluation set of 200 customer interactions with zero hardcoded metrics.
6. **Provides Multiple Interfaces** including an interactive Amazon-themed Web UI and a command-line interface (CLI).

---

## 🏗️ System Architecture

```
                      [ Incoming Customer Message ]
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ 1. Pre-LLM Safety Filter (Regex / Keyword Triggers)      │
       │    Detects legal, security, safety, and fraud keywords    │
       └────────────────────────────┬─────────────────────────────┘
                                    │ No risk detected
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ 2. LLM Intent Classifier (Nemotron-3-Super-120B)         │
       │    Assigns 1 of 8 intents + numerical confidence score   │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ 3. Historical Case Retrieval Engine (TF-IDF RAG)         │
       │    Retrieves top matching resolved AmazonHelp cases      │
       │    (Strictly excludes any held-out evaluation threads)   │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ 4. Response Generation (Grounded LLM Prompting)          │
       │    Drafts empathetic, policy-compliant support reply     │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ 5. Post-Generation Routing Policy                        │
       │    • If intent = other_unclear or account_issue ──► ESCALATE │
       │    • If confidence < 0.60 ────────────────────────► ESCALATE │
       │    • Otherwise ───────────────────────────────────► AUTO     │
       └──────────────────────────────────────────────────────────┘
```

---

## 🏷️ 8-Class Intent Taxonomy

The intent taxonomy was derived directly from the Kaggle *Customer Support on Twitter* dataset (`twcs.csv`) through iterative batch analysis:

| Intent | Description | Typical Example | Default Routing |
| :--- | :--- | :--- | :---: |
| `shipping_delivery` | Package transit, late delivery, courier tracking | *"My parcel is 3 days late, tracking not updating"* | `AUTO` |
| `return_refund` | Return requests, replacement items, refund status | *"Received broken item, want a replacement"* | `AUTO` |
| `payment_issue` | Billing disputes, double charges, Prime fee queries | *"Why was I charged twice on my card for Prime?"* | `AUTO` |
| `order_status` | Order placement confirmation, order modification | *"Where is my order #123-4567890-1234567?"* | `AUTO` |
| `product_question` | Item specifications, pricing, stock availability | *"When will the Echo Dot 4th Gen be back in stock?"* | `AUTO` |
| `account_issue` | Login problems, password reset, account lockouts | *"I cannot access my account after password reset"* | `ESCALATE` |
| `technical_support` | App crashes, website errors, Kindle device bugs | *"Amazon app crashes whenever I tap checkout"* | `AUTO` |
| `other_unclear` | Ambiguous queries, general complaints, banter | *"You guys are terrible, nothing works"* | `ESCALATE` |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- An NVIDIA API Key (OpenAI-compatible endpoint from [build.nvidia.com](https://build.nvidia.com/))

### Steps

```bash
# 1. Navigate to the project directory
cd /home/harsha/Desktop/customer

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Configure your NVIDIA API Key
export NVIDIA_API_KEY="your-nvidia-api-key-here"
```

---

## 🖥️ How to Run & Test the System

You can test and demonstrate the agent using **two different interfaces**:

### Option 1: Interactive Amazon Web UI (Recommended for Demo) 🌐

An interactive web portal styled after the Amazon desktop store with an embedded floating AI support chatbot.

1. **Start the Web Server:**
   ```bash
   python frontend/server.py
   ```
2. **Open in Browser:**
   Navigate to **`http://localhost:5000`**
3. **Interact with the Agent:**
   - Click the yellow **"Amazon Support AI"** button at the bottom-right.
   - Type any customer support inquiry or click the pre-configured prompt chips.
   - The agent responds with a grounded support reply, along with live diagnostic tags showing the **classified intent**, **confidence score**, **routing decision** (`AUTO` vs `ESCALATE`), and **retrieved RAG case IDs**.

---

### Option 2: Command-Line Interface (CLI) 💻

Test individual inquiries interactively from the terminal:

```bash
python -m src.reply_agent --interactive
```

**Example CLI Interaction:**
```text
============================================================
🤖 AmazonHelp AI Support Agent — Interactive CLI Preview
============================================================
Type a customer message to preview agent classification, grounding RAG, reply, and routing.
Type 'exit' or 'quit' to exit.

User Inquiry: I received a damaged coffee maker and want my money back
🔄 Processing inquiry...
------------------------------------------------------------
📌 Classified Intent : return_refund (Confidence: 0.95)
🚦 Routing Decision  : AUTO (High-confidence automated resolution available for intent 'return_refund'.)
📚 Grounding Case IDs: ['1272355', '2599777']
💬 Drafted Response  :
I’m really sorry to hear that your coffee maker arrived damaged! You can start a return or request a refund by visiting 'Your Orders' and selecting 'Return or replace items'. If you need any assistance with the process, please send us a Direct Message with your order number so we can help directly.
------------------------------------------------------------
```

---

## 📊 Evaluation & Benchmark Results

The system was evaluated against a **held-out golden evaluation set of 200 customer inquiries** with ground-truth human annotations. Metrics were calculated with **zero hardcoding or artificial inflation**.

### 1. Classification & Routing Performance

| Metric | Trivial Baseline | Simple Rule Baseline | Nemotron RAG Agent | What This Means |
| :--- | :---: | :---: | :---: | :--- |
| **Intent Accuracy** | 22.5% | **48.0%** | 40.5% | Rule matching performs well on short tweets with obvious keywords; LLM handles nuanced language. |
| **Macro F1** | 0.046 | **0.443** | 0.340 | Balances performance across rare and frequent intents. |
| **Routing Accuracy** | 22.5% | **70.5%** | 65.5% | Percentage of correct `AUTO` vs `ESCALATE` decisions. |
| **Escalation Recall** | **100.0%** | 53.3% | 44.4% | Proportion of human-escalated cases correctly caught. |
| **False Auto-Handle Rate** | **0.0%** | 46.7% | 55.6% | **Critical Safety Metric:** Frustrated/sensitive cases wrongly sent to automated reply. |

### 2. Reply Quality (LLM-as-a-Judge, 1–5 Scale)

An independent LLM judge evaluated generated replies across 5 dimensions:

| Quality Dimension | Score | Description |
| :--- | :---: | :--- |
| **Correctness** | **4.97 / 5.0** | Factual accuracy without hallucinating delivery dates or refund promises |
| **Groundedness** | **4.97 / 5.0** | Strict alignment with Amazon corporate support guidelines |
| **Relevance** | **4.83 / 5.0** | Directly addresses the customer's specific problem |
| **Completeness** | **4.69 / 5.0** | Provides clear next steps (e.g., DM order ID, check 'Your Orders') |
| **Tone & Empathy** | **4.94 / 5.0** | Empathetic, polite, and professional customer brand voice |
| **Overall Score** | **4.88 / 5.0** | High-quality, safe, and actionable customer replies |

### 3. Data Leakage Prevention
To guarantee scientific integrity:
- The 200 evaluation threads are strictly excluded from the retrieval corpus.
- The TF-IDF retrieval index uses 4,787 distinct historical cases (`knowledge_corpus.jsonl`).
- An automated assertion verified that 0% of evaluation queries retrieved their own thread or ground-truth reply.

---

## 🔍 Key Engineering Insights & Failure Analysis

In-depth error analysis revealed important operational tradeoffs:

1. **The Keyword vs. LLM Accuracy Dynamic:**
   On short Twitter posts (<20 words), keyword matching achieves slightly higher single-label accuracy (48.0% vs 40.5%) because users often type direct words like *"refund"* or *"return"*. The LLM classifier provides deeper semantic understanding but can over-analyze ambiguous messages.
2. **False Auto-Handle as Primary SLA:**
   In customer support, sending an automated canned reply to an angry customer demanding human assistance causes severe customer churn. Addressing this requires sentiment-aware escalation rules.
3. **High Response Quality Across Edge Cases:**
   Even when an intent was subtly misclassified (e.g., classifying a delivery issue as general order status), the generated reply scored **4.88 / 5.0** because the LLM drafts helpful, empathetic instructions asking for order details via DM regardless of minor label taxonomy overlaps.

---

## 📁 Repository Structure

```
├── dataset/                    # Source dataset directory (twcs.csv)
├── data/
│   ├── processed/              # Cleaned threads, knowledge_corpus.jsonl, taxonomy.json
│   ├── eval/                   # 200 human-annotated golden examples & SAMPLING_NOTES.md
│   └── results/                # Evaluation logs, metrics_summary.json, llm_judge.jsonl
├── frontend/                   # Interactive Amazon Web Portal & Chatbot UI
│   ├── index.html              # Amazon clone web page
│   ├── style.css               # Amazon Ember design system stylesheet
│   ├── script.js               # Chatbot client logic & diagnostic badges
│   └── server.py               # Flask backend connecting UI to AI agent pipeline
├── src/
│   ├── llm_client.py           # NVIDIA Nemotron API client with JSON parsing & exponential backoff
│   ├── ingest.py               # Multi-turn thread reconstruction from raw CSV
│   ├── clean.py                # Regex PII scrubbing & text normalization
│   ├── taxonomy.py             # 8-class intent taxonomy synthesis
│   ├── create_human_review.py  # Golden set sampler & annotation scaffold
│   ├── baselines.py            # Trivial and Simple rule-based baseline implementations
│   ├── classifier.py           # Intent classifier with numerical confidence scoring
│   ├── retrieval.py            # TF-IDF RAG retrieval engine over 4,787 resolved cases
│   ├── reply_agent.py          # Complete agent pipeline & multi-tier routing logic
│   └── evaluate.py             # Automated metrics computation & LLM-as-a-judge scoring
├── report/
│   ├── DECISION_LOG.md         # 14 documented engineering decisions & tradeoffs
│   └── REPORT.md               # In-depth system report, error analysis, and roadmap
├── CREDITS.md                  # Attribution for datasets and open-source libraries
└── README.md                   # Project documentation & execution guide
```

---
## Conclusion
# Why I Built This

This project was built as an engineering exercise to explore how an AI support agent can combine intent classification, historical-case retrieval, response generation, and human-escalation decisions in a single workflow.

The goal was not just to generate plausible support replies, but to build an evaluation-driven system where each major component can be measured against a human-labelled ground-truth set.

The project therefore focuses on three questions:

1. **Can the agent correctly understand the customer's intent?**
2. **Can it generate a useful response grounded in how similar cases were handled historically?**
3. **Can it reliably distinguish cases that can be handled automatically from those that should be escalated to a human?**

The evaluation intentionally uses held-out examples and human-reviewed labels, with separate baselines and failure analysis, so that the reported results reflect the actual behaviour of the system rather than an optimistic demonstration.

## Limitations

This is a prototype built for the Hiver SDE Intern take-home assignment. The evaluation dataset is a relatively small human-labelled sample, and the system's routing confidence is not a substitute for production-grade calibration or monitoring.

The results should therefore be interpreted as an evaluation of this prototype on the defined test set, not as a claim of production readiness.

## Future Improvements

With additional development time, I would focus on:

* Calibrating routing confidence using a held-out validation set.
* Improving handling of ambiguous and multi-intent customer messages.
* Strengthening retrieval quality and evidence selection.
* Improving escalation detection for repeated or unresolved support interactions.
* Expanding the human-labelled evaluation set.
* Adding production-style monitoring for incorrect auto-handling and escalation decisions.

---

**Built as part of the Hiver SDE Intern take-home assignment.**

