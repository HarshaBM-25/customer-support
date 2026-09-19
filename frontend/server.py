import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.reply_agent import AmazonSupportAgent

app = Flask(__name__, static_folder=".")
CORS(app)

# Initialize agent once on startup
print("Initializing Amazon Support Agent for web frontend...")
agent = AmazonSupportAgent()
print("✓ Support Agent initialized and ready.")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        result = agent.process_message(message)
        return jsonify({
            "status": "success",
            "intent": result["intent"],
            "confidence": round(float(result["confidence"]), 2),
            "route": result["route"],
            "route_reason": result["route_reason"],
            "retrieved_cases": result["retrieved_cases"],
            "reply": result["draft_reply"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "reply": "I apologize, but I encountered a system issue while looking up your inquiry. Please reach out to customer support via Direct Message with your order ID.",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"🚀 Amazon Help AI Agent Web UI running at:")
    print(f"   http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
