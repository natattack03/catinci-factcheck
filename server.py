"""
server.py – Catinci Fact-Check API (Live Version)
-------------------------------------------------
Receives a claim, searches Google for evidence, sends it to Gemini for reasoning,
and returns a structured JSON verdict (true, false, or unsure) with citations and spoken text.
"""

import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from urllib.parse import urlparse

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

# Preferred domains and penalties for ranking search results
TRUSTED_DOMAIN_PATTERNS = [
    ("nasa.gov", 400),
    ("nationalgeographic.com", 350),
    ("kids.nationalgeographic.com", 350),
    ("britannica.com", 350),
    ("smithsonianmag.com", 325),
    ("who.int", 325),
    ("nih.gov", 325),
    ("cdc.gov", 325),
    ("noaa.gov", 300),
    ("mit.edu", 300),
    ("harvard.edu", 300),
    ("stanford.edu", 300),
    ("edu", 0),  # generic fallback; bonus handled by TLD score
]

LOW_CREDIBILITY_DOMAINS = [
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "fandom.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "medium.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "instagram.com",
]


def normalize_domain(url: str) -> str:
    """Extract base domain from a URL."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def domain_priority_score(domain: str) -> int:
    """Score domains to encourage educational and scientific sources."""
    if not domain:
        return 0

    # Penalize low-credibility domains heavily
    if any(domain.endswith(bad) or bad in domain for bad in LOW_CREDIBILITY_DOMAINS):
        return -200

    score = 0
    if domain.endswith(".edu"):
        score += 400
    elif domain.endswith(".gov"):
        score += 350
    elif domain.endswith(".org"):
        score += 250
    elif domain.endswith(".com"):
        score += 150
    else:
        score += 100

    for pattern, bonus in TRUSTED_DOMAIN_PATTERNS:
        if domain.endswith(pattern):
            score += bonus
            break

    return score

# -------------------------------------------------------------------
# 2. Helper: Format spoken response for ElevenLabs
# -------------------------------------------------------------------
def format_spoken_response(verdict, rationale, citations):
    verdict_text = {
        "true": "That’s true —",
        "false": "That’s not true —",
        "unsure": "I’m not completely sure —"
    }.get(verdict.lower(), "I’m not completely sure —")

    # Clean rationale (remove markdown, code blocks, extra spaces)
    if rationale:
        clean_rationale = re.sub(r"```.*?```", "", rationale, flags=re.DOTALL)
        clean_rationale = re.sub(r"\*\*.*?\*\*", "", clean_rationale)  # remove bold markers like **false**
        clean_rationale = re.sub(r"\s+", " ", clean_rationale).strip()

        # Extract a meaningful sentence
        sentences = re.split(r"(?<=[.!?])\s+", clean_rationale)
        short_reason = next((s for s in sentences if len(s.split()) > 4 and not s.lower().startswith(("true", "false"))), sentences[0])
        # Make sure it ends cleanly with punctuation
        if not short_reason.endswith(('.', '!', '?')):
            short_reason += '.'
    else:
        short_reason = ""

    # Prefer reputable sources for spoken citation
    source = None
    preferred_domains = ["wikipedia", "nasa", "nationalgeographic", "bbc", "scientificamerican", "mit.edu"]
    if citations and isinstance(citations, list):
        for c in citations:
            url = c.get("url", "").lower()
            title = c.get("title", "")
            if any(domain in url for domain in preferred_domains):
                source = title.split(" | ")[0].split(" - ")[0]
                break
        if not source and len(citations) > 0:
            source = citations[0].get("title", "").split(" | ")[0].split(" - ")[0]

    # Construct the spoken sentence
    if source:
        return f"{verdict_text} According to {source}, {short_reason}"
    else:
        return f"{verdict_text} {short_reason}"




# -------------------------------------------------------------------
# 3. Core fact-check logic
# -------------------------------------------------------------------
def run_fact_check_logic(query: str):
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
            "citations": [],
            "spoken": "I’m not completely sure — There was a problem accessing the fact-check data."
        }

    if not items:
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": "No search results found for this query.",
            "citations": [],
            "spoken": "I’m not completely sure — I couldn’t find enough information to answer that."
        }

    annotated_items = []
    for idx, item in enumerate(items):
        link = item.get("link", "")
        domain = normalize_domain(link)
        score = domain_priority_score(domain)
        annotated_items.append(
            {
                "rank": idx,
                "item": item,
                "domain": domain,
                "score": score,
            }
        )

    sorted_items = sorted(annotated_items, key=lambda entry: (-entry["score"], entry["rank"]))

    evidence_lines = []
    for entry in sorted_items:
        title = (entry["item"].get("title") or "").strip()
        snippet = (entry["item"].get("snippet") or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        domain_label = entry["domain"] or "unknown"
        evidence_lines.append(f"- {domain_label}: {title}: {snippet}")

    evidence = "\n".join(evidence_lines)

    # --- Gemini reasoning phase ---
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
Claim: {query}

Evidence (from Google search snippets):
{evidence}

Based on the evidence above, classify the claim as one of:
- true (supported by reputable evidence)
- false (contradicted by reputable evidence)
- unsure (unclear or insufficient evidence)

Then, write your response in this format:
"That’s [true/false/unsure] — According to [source name], [one-sentence explanation in simple words that a 5–7 year old can understand]."

Rules:
- Pick one clear, trustworthy source. Prioritize sources by domain: .edu > .gov > .org > .com > other.
- Prefer well-known scientific or educational organizations (NASA, National Geographic, Britannica, Smithsonian, WHO, NIH, universities).
- Avoid user-generated or crowd-based sites (Reddit, Quora, forums, fandom wikis) unless no other reputable information is available.
- State the source using its domain name (e.g., "According to nasa.gov").
- Avoid technical words like “snippet,” “rationale,” or “evidence.”
- Keep the tone short, warm, and easy to understand.
- Do not return JSON or code blocks.
	"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        print(f"🤖 Gemini raw output:\n{text}\n")

        verdict = "unsure"
        if "true" in text.lower() and "not true" not in text.lower():
            verdict = "true"
        elif any(word in text.lower() for word in ["false", "incorrect", "not true"]):
            verdict = "false"

        result = {
            "verdict": verdict,
            "confidence": 0.8,
            "rationale": text,
            "citations": [
                {
                    "title": entry["item"].get("title", ""),
                    "url": entry["item"].get("link", ""),
                    "quote": entry["item"].get("snippet", ""),
                }
                for entry in sorted_items[:3]
            ]
        }

        # Add spoken summary
        result["spoken"] = format_spoken_response(result["verdict"], result["rationale"], result["citations"])
        return result

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": f"Gemini API error: {e}",
            "citations": [],
            "spoken": "I’m not completely sure — I couldn’t reach the reasoning service."
        }


# -------------------------------------------------------------------
# 4. API routes
# -------------------------------------------------------------------
@app.route("/fact_check", methods=["POST"])
def fact_check():
    try:
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"spoken": "I couldn’t read that fact to check it. Please send it again."}), 200

        claim = data.get("claim", "")
        if not isinstance(claim, str):
            return jsonify({"spoken": "I only understand facts written as text sentences."}), 200
        claim = claim.strip()
        print(f"🧠 Received claim: {claim}")

        if not claim:
            return jsonify({"spoken": "I didn’t catch that to fact-check, sorry!"}), 200

        result = run_fact_check_logic(claim)
        print("🤖 Gemini raw output:", result.get("rationale", ""))

        # 🧩 Extract the short summary to speak aloud
        # Prefer Gemini’s explanation; fallback to rationale text
        explanation = result.get("rationale", "").strip()
        spoken = explanation

        # 📝 Append detailed log entry with sources for review
        try:
            with open("factcheck_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n🕓 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    f"\n🔎 Claim: {claim}"
                    f"\n🗣️ Agent would say (full): {spoken}"
                )

                citations = result.get("citations") or []
                if citations:
                    log_file.write("\n📚 Sources:")
                    for cite in citations:
                        title = (cite.get("title") or "").strip()
                        url = (cite.get("url") or "").strip()
                        if title or url:
                            log_file.write(f"\n  • {title} — {url}")
                log_file.write(f"\n{'-'*80}\n")
        except Exception as log_error:
            print(f"⚠️ Could not write to factcheck_log.txt: {log_error}")

        # Ensure ElevenLabs gets *only* what it can read aloud
        return jsonify({"spoken": spoken}), 200

    except Exception as e:
        print(f"❌ Error in /fact_check: {e}")
        return jsonify({"spoken": "Hmm, my fact-checking tool had trouble just now!"}), 200




@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
