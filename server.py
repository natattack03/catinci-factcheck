"""
server.py – Catinci Fact-Check API (Live Version)
-------------------------------------------------
Receives a claim, searches Google for evidence, sends it to Gemini for reasoning,
and returns a structured JSON verdict (true, false, or unsure) with citations.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------------------------------------------------
# 1. Setup
# -------------------------------------------------------------------
load_dotenv()
app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)


# -------------------------------------------------------------------
# 2. Fact-check core logic
# -------------------------------------------------------------------
def run_fact_check_logic(query: str):
    """
    1️⃣ Search Google Custom Search for snippets related to the claim
    2️⃣ Ask Gemini to reason over the snippets
    3️⃣ Return verdict, rationale, and top citations
    """
    print(f"🧠 Running live fact-check logic for: {query}")

    # --- Google Search phase ---
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CX_ID, num=5).execute()
        items = res.get("items", [])
    except Exception as e:
        print(f"❌ Google search error: {e}")
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": f"Google Search error: {e}",
            "citations": []
        }

    if not items:
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": "No search results found for this query.",
            "citations": []
        }

    # Combine snippets into evidence text
    evidence = "\n".join(f"{i['title']}: {i['snippet']}" for i in items)

    # --- Gemini reasoning phase ---
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
Claim: {query}

Evidence (from Google search snippets):
{evidence}

Based on the evidence above, classify the claim as one of:
- true  (supported by reputable evidence)
- false (contradicted by reputable evidence)
- unsure (unclear or insufficient evidence)

Return a JSON object with fields:
verdict, confidence (0–1), rationale.
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        print(f"🤖 Gemini raw output:\n{text}\n")

        # Extract verdict from Gemini’s text
        verdict = "unsure"
        if "true" in text.lower():
            verdict = "true"
        elif "false" in text.lower():
            verdict = "false"

        # --- Build structured result ---
        return {
            "verdict": verdict,
            "confidence": 0.8,
            "rationale": text,
            "citations": [
                {"title": i["title"], "url": i["link"], "quote": i["snippet"]}
                for i in items[:3]
            ]
        }

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": f"Gemini API error: {e}",
            "citations": []
        }


# -------------------------------------------------------------------
# 3. API route
# -------------------------------------------------------------------
@app.route("/fact_check", methods=["POST"])
def fact_check():
    data = request.get_json(force=True, silent=True)
    claim = None

    if data and "claim" in data:
        claim = data["claim"].strip()
    elif request.json and "claim" in request.json:
        claim = request.json["claim"].strip()

    print(f"🧠 Received claim: {claim}")

    if not claim:
        return jsonify({
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": "No claim text provided in request.",
            "citations": []
        })



# -------------------------------------------------------------------
# 4. Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print(f"🚀 Starting Fact-Check server on http://127.0.0.1:{port}/fact_check")
    app.run(host="0.0.0.0", port=port)
