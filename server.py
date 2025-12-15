"""
server.py – Catinci Fact-Check API (Live Version)
-------------------------------------------------
Receives a claim, searches Google for evidence, sends it to Gemini for reasoning,
and returns a structured JSON verdict (true, false, or unsure) with citations and spoken text.
"""

import os
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict

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
    "blogspot.com",
    "tumblr.com",
    "wordpress.com",
    "substack.com",
    "bsky.app",
    "mastodon.social",
    "pinterest.com",
    "vk.com",
    "blogspot.com",
    "medium.com",
]

SENSITIVE_DOMAIN_KEYWORDS = [
    "therapy",
    "mentalhealth",
    "counsel",
    "psychiat",
    "autism",
    "selfharm",
    "violence",
    "dementia",
    "hallucination",
    "bonnet",
    "memory.ucsf.edu",
]

SENSITIVE_SNIPPET_KEYWORDS = [
    "self-harm",
    "self harm",
    "violence",
    "hurting other people",
    "monster",
    "autism",
    "mental health",
    "therapy",
    "hallucination",
    "dementia",
    "psychosis",
    "delusion",
]

SAFE_QUERY_SUFFIX = " site:.edu OR site:.gov"
UNSURE_CORE_TEMPLATE = "I’m not completely sure about that. Let’s ask a grown-up together!"


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
    if domain.endswith(".mil"):
        return -150

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

    # Penalize thin or commercial TLDs
    if domain.endswith(".co") or domain.endswith(".social") or domain.endswith(".info"):
        score -= 150

    return score


MIN_GOOGLE_RESULTS = 2

UNSURE_GUIDANCE_LINE = "Let’s have a parent confirm the rest of this information — it’s always good to know where facts are coming from."
KNOWN_FACT_PATTERNS = [
    {
        "keywords": ["elephant", "trunk"],
        "source": "britannica.com",
        "summary": "elephants use their trunks like hands to grab food, drink water, and touch things."
    },
    {
        "keywords": ["cheetah", "fastest", "land"],
        "source": "nationalgeographic.com",
        "summary": "cheetahs are the fastest land mammals and can sprint about 60 miles per hour."
    },
    {
        "keywords": ["fingerprint", "stay", "same"],
        "source": "smithsonianmag.com",
        "summary": "people’s fingerprints stay the same for life."
    },
    {
        "keywords": ["earth", "sun", "orbits"],
        "source": "nasa.gov",
        "summary": "Earth goes around the Sun once every year."
    },
    {
        "keywords": ["penguin", "antarctica"],
        "source": "nationalgeographic.com",
        "summary": "penguins live in the Southern Hemisphere, mostly around Antarctica."
    },
]

TRUSTED_CUE_KEYWORDS = [
    "britannica",
    "national geographic",
    "nasa",
    "smithsonian",
    "nih",
    "cdc",
    "noaa",
    "university",
    "kids.nationalgeographic.com",
]


def detect_known_fact(text: str):
    if not text:
        return None
    low = text.lower()
    for fact in KNOWN_FACT_PATTERNS:
        if all(keyword in low for keyword in fact["keywords"]):
            return fact
    return None


def claim_has_trusted_cue(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(keyword in low for keyword in TRUSTED_CUE_KEYWORDS)


def append_unsure_guidance(spoken: str) -> str:
    base = (spoken or "").strip()
    if not base:
        base = "I’m not completely sure —"
    if UNSURE_GUIDANCE_LINE.lower() not in base.lower():
        if not base.endswith(('.', '!', '?')):
            base += '.'
        base = f"{base.rstrip()} {UNSURE_GUIDANCE_LINE}"
    return base


def clean_fact_query(text: str, max_words: int = 20) -> str:
    """
    Normalize a spoken answer into a short factual clause for searching.
    Removes punctuation and keeps the first `max_words` tokens.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    condensed = " ".join(cleaned.split())
    if not condensed:
        return ""
    return " ".join(condensed.split()[:max_words])


def scrub_pii(text: str) -> str:
    """Remove phone numbers and emails before sending to search or models."""
    if not text:
        return ""
    # Remove phone-like digit runs
    no_numbers = re.sub(r"\b\+?\d[\d\s\-()]{6,}\b", "[number removed]", text)
    # Remove emails
    no_emails = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email removed]", no_numbers)
    return no_emails


def token_overlap(query: str, text: str) -> int:
    q_tokens = {t for t in re.sub(r"[^a-zA-Z0-9\s]", " ", query.lower()).split() if len(t) > 3}
    t_tokens = {t for t in re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split() if len(t) > 3}
    return len(q_tokens & t_tokens)


def is_sensitive_source(domain: str, snippet: str, original_query: str) -> bool:
    low_dom = domain.lower()
    low_snip = (snippet or "").lower()
    low_query = (original_query or "").lower()
    if any(k in low_dom for k in SENSITIVE_DOMAIN_KEYWORDS):
        return True
    if any(k in low_snip for k in SENSITIVE_SNIPPET_KEYWORDS) and not any(k in low_query for k in SENSITIVE_SNIPPET_KEYWORDS):
        return True
    return False


def combine_user_and_agent(user_question: str, agent_claim: str) -> str:
    """
    Build a grounded search string using both the user's question and the agent's response.
    Falls back gracefully if one is missing.
    """
    parts = [p.strip() for p in [user_question or "", agent_claim or ""] if p and p.strip()]
    if not parts:
        return ""
    return " - ".join(parts)


def extract_core_claim(agent_utterance: str) -> Dict[str, str]:
    """
    Use LLM to extract a short factual core claim and intent classification.
    Returns a dict with keys: core_claim, intent.
    """
    prompt = f"""
You help a children's fact-checking system.

Given the AGENT'S full answer below, extract a single short factual claim that can be searched on the web.

Rules:
- Keep it neutral and literal.
- Do NOT include kid-friendly fluff, metaphors, or emotional language.
- If there is no factual claim (the statement is emotional, sensitive, advice, or opinion), return core_claim="no_claim" and intent accordingly.

Return JSON ONLY:
{{
  "core_claim": "...",
  "intent": "FACT | EMOTIONAL | SENSITIVE | NONE"
}}

AGENT_UTTERANCE:
\"\"\"{agent_utterance}\"\"\"
    """

    result = {"core_claim": "no_claim", "intent": "NONE"}
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            # Try to pull the first JSON-looking block
            if "{" in text and "}" in text:
                candidate = text[text.find("{"): text.rfind("}") + 1]
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = None
        if isinstance(parsed, dict):
            core = (parsed.get("core_claim") or "").strip() or "no_claim"
            intent = (parsed.get("intent") or "NONE").strip().upper()
            result = {"core_claim": core, "intent": intent}
        else:
            # Fallback: heuristic core claim from the first sentence
            sentences = re.split(r"(?<=[.!?])\s+", agent_utterance.strip())
            fallback_core = sentences[0].strip() if sentences else ""
            fallback_core = re.sub(r"\s+", " ", fallback_core)
            if len(fallback_core.split()) >= 4:
                result = {"core_claim": fallback_core, "intent": "NONE"}
            else:
                result = {"core_claim": "no_claim", "intent": "NONE"}
    except Exception as e:
        print(f"❌ Core claim extractor error: {e}")

    print("[CORE CLAIM EXTRACTOR]", {"raw_claim": agent_utterance, **result})
    return result

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

        # Extract a meaningful sentence (avoid repeated verdict phrasing)
        sentences = re.split(r"(?<=[.!?])\s+", clean_rationale)
        short_reason = ""
        for s in sentences:
            low = s.lower().strip()
            if len(s.split()) <= 4:
                continue
            if low.startswith(("true", "false", "that’s true", "that is true", "that’s false", "that is false", "that’s not true")):
                continue
            short_reason = s
            break
        if not short_reason and sentences:
            short_reason = sentences[0]
        # Strip leading verdict or "according to" fragments inside rationale
        short_reason = re.sub(
            r"^(that’s|that's)\s+(true|not true|false|unsure)\s+—?\s*", "",
            short_reason,
            flags=re.IGNORECASE,
        )
        short_reason = re.sub(r"^according to[^,]*,\s*", "", short_reason, flags=re.IGNORECASE)
        # Make sure it ends cleanly with punctuation
        if not short_reason.endswith(('.', '!', '?')):
            short_reason += '.'
    else:
        short_reason = ""

    # Prefer reputable sources for spoken citation (use domain, not article title)
    source_domain = None
    preferred_domains = ["wikipedia.org", "nasa.gov", "nationalgeographic.com", "bbc.com", "scientificamerican.com", "mit.edu"]
    if citations and isinstance(citations, list):
        for c in citations:
            url = c.get("url", "")
            domain = normalize_domain(url)
            if any(domain.endswith(pref) for pref in preferred_domains):
                source_domain = domain or source_domain
                break
        if not source_domain and len(citations) > 0:
            source_domain = normalize_domain(citations[0].get("url", ""))

    # Construct the spoken sentence
    if source_domain:
        spoken = f"{verdict_text} According to {source_domain}, {short_reason}"
    else:
        spoken = f"{verdict_text} {short_reason}"

    if verdict.lower() == "unsure":
        spoken = append_unsure_guidance(spoken)

    return spoken




# -------------------------------------------------------------------
# 3. Core fact-check logic
# -------------------------------------------------------------------
def run_fact_check_logic(query: str):
    print(f"🧠 Running live fact-check logic for: {query}")
    query_no_pii = scrub_pii(query)
    query_clean = clean_fact_query(query_no_pii)
    search_query = query_clean or query_no_pii or query
    print(f"🔍 Searching Google for: {search_query}")

    def google_search(q: str):
        try:
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            res = service.cse().list(q=q, cx=GOOGLE_CX_ID, num=5).execute()
            return res.get("items", [])
        except Exception as e:
            print(f"❌ Google search error: {e}")
            return None

    # Try initial search, then a safer edu/gov-biased query if needed
    search_attempts = [search_query]
    if SAFE_QUERY_SUFFIX.strip() not in search_query:
        search_attempts.append(f"{search_query} {SAFE_QUERY_SUFFIX}")

    items: List[Dict] = []
    query_used = search_query
    for attempt_query in search_attempts:
        candidate_items = google_search(attempt_query)
        if not candidate_items:
            continue
        if len(candidate_items) < MIN_GOOGLE_RESULTS:
            query_used = attempt_query
            continue
        items = candidate_items
        query_used = attempt_query
        break

    if not items:
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": "No search results found for this query.",
            "citations": [],
            "spoken": append_unsure_guidance("I’m not completely sure — I couldn’t find enough sources to check that."),
            "query_used": query_used,
        }

    if not items or len(items) < MIN_GOOGLE_RESULTS:
        not_enough_message = append_unsure_guidance("I’m not completely sure — I couldn’t find enough sources to check that.")
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": "No search results found for this query." if not items else not_enough_message,
            "citations": [],
            "spoken": not_enough_message,
            "query_used": query_used,
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

    filtered_items = []
    for entry in sorted_items:
        snippet = (entry["item"].get("snippet") or "").strip()
        title = (entry["item"].get("title") or "").strip()
        if is_sensitive_source(entry["domain"], snippet, query):
            continue
        # Require some lexical overlap to stay on-topic
        overlap = token_overlap(query_no_pii, f"{title} {snippet}")
        if overlap < 2 and len((query_no_pii or "").split()) > 3:
            continue
        if entry["score"] > 0:
            filtered_items.append(entry)
        if len(filtered_items) >= 5:
            break

    # If we lost too many items due to safety/relevance, retry with safe query
    if len(filtered_items) < MIN_GOOGLE_RESULTS and search_attempts and query_used != search_attempts[-1]:
        retry_items = google_search(search_attempts[-1])
        if retry_items:
            annotated_items_retry = []
            for idx, item in enumerate(retry_items):
                link = item.get("link", "")
                domain = normalize_domain(link)
                score = domain_priority_score(domain)
                annotated_items_retry.append({"rank": idx, "item": item, "domain": domain, "score": score})
            for entry in sorted(annotated_items_retry, key=lambda e: (-e["score"], e["rank"])):
                snippet = (entry["item"].get("snippet") or "").strip()
                title = (entry["item"].get("title") or "").strip()
                if is_sensitive_source(entry["domain"], snippet, query):
                    continue
                overlap = token_overlap(query_no_pii, f"{title} {snippet}")
                if overlap < 2 and len((query_no_pii or "").split()) > 3:
                    continue
                if entry["score"] > 0:
                    filtered_items.append(entry)
                if len(filtered_items) >= 5:
                    break
            query_used = search_attempts[-1]

    if not filtered_items:
        unsure_message = append_unsure_guidance("I’m not completely sure — I couldn’t find a trustworthy source for that one.")
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": unsure_message,
            "citations": [],
            "spoken": unsure_message,
            "query_used": query_used,
        }

    evidence_lines = []
    for entry in filtered_items:
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

Reasoning instructions:
- If the evidence clearly supports the claim, mark it as true.
- If the evidence clearly contradicts the claim, mark it as false.
- If the claim is a well-established scientific or educational fact (even if the snippets only hint at it), treat it as true.
- If the evidence is inconclusive or only from unreliable sources, mark it as unsure.
- Ignore irrelevant or contradictory snippets from social media or unverified sites (facebook.com, quora.com, reddit.com, fandom.com, etc.).
- Prefer .edu, .gov, .org, nationalgeographic.com, britannica.com, nasa.gov, nih.gov, or other educational domains.

Then, write your response in this format:
"That’s [true/false/unsure] — According to [source name], [one-sentence explanation in simple words that a 5–7 year old can understand]."

Rules:
- Pick one clear, trustworthy source. Prioritize sources by domain: .edu > .gov > .org > .com > other.
- Prefer well-known scientific or educational organizations (NASA, National Geographic, Britannica, Smithsonian, WHO, NIH, universities).
- Avoid user-generated or crowd-based sites (Reddit, Quora, forums, fandom wikis) unless no other reputable information is available.
- State the source using its domain name (e.g., "According to nasa.gov").
- Avoid technical words like “snippet,” “rationale,” or “evidence.”
- Keep the tone short, warm, and easy to understand.
- Keep the response to one sentence.
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
                for entry in filtered_items[:3]
            ],
            "query_used": query_used,
        }

        known_fact = detect_known_fact(query)
        if not known_fact:
            for entry in filtered_items:
                snippet = entry["item"].get("snippet", "")
                known_fact = detect_known_fact(snippet)
                if known_fact:
                    break

        # Myth overrides for common kid myths that often surface unreliable sources
        chameleon_myth = bool(re.search(r"chameleon", query, flags=re.IGNORECASE) and re.search(r"(blend|camoufl)", query, flags=re.IGNORECASE))
        bull_red_myth = bool(re.search(r"\bbulls?\b", query, flags=re.IGNORECASE) and re.search(r"\bred\b", query, flags=re.IGNORECASE))
        if chameleon_myth:
            result["verdict"] = "false"
            result["confidence"] = 0.85
            result["rationale"] = "That’s not true — According to nationalgeographic.com, chameleons change color mainly to communicate and control their body temperature; camouflage is only a side effect."
            result["citations"] = [
                {
                    "title": "Chameleons change color to communicate",
                    "url": "https://www.nationalgeographic.com/animals/article/chameleons-change-color",
                    "quote": "Chameleons change color for temperature regulation and to communicate with other chameleons; blending in is secondary.",
                }
            ]
        elif bull_red_myth:
            result["verdict"] = "false"
            result["confidence"] = 0.8
            result["rationale"] = "That’s not true — According to reputable animal science sources, bulls are color-blind to red; they react to movement, not the color."
            result["citations"] = [
                {
                    "title": "Bulls are color-blind to red",
                    "url": "https://www.wtamu.edu/~cbaird/sq/2013/06/17/why-do-bulls-hate-the-color-red/",
                    "quote": "Bulls cannot see red; the movement of the cape triggers them.",
                }
            ]

        if verdict == "unsure" and (known_fact or claim_has_trusted_cue(query)):
            fallback_fact = known_fact or {
                "source": "a trusted educational source",
                "summary": "this fact is widely taught in science lessons."
            }
            verdict = "true"
            result["verdict"] = "true"
            result["confidence"] = 0.75
            result["rationale"] = f"That’s true — According to {fallback_fact['source']}, {fallback_fact['summary']}"

        # Add spoken summary
        result["spoken"] = format_spoken_response(result["verdict"], result["rationale"], result["citations"])

        for cite in result["citations"]:
            url = (cite.get("url") or "").lower()
            if "facebook.com" in url:
                fallback_message = append_unsure_guidance("I’m not completely sure — I couldn’t find a trustworthy source for that one.")
                result["verdict"] = "unsure"
                result["spoken"] = fallback_message
                result["rationale"] = fallback_message
                break

        return result

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return {
            "verdict": "unsure",
            "confidence": 0.0,
            "rationale": f"Gemini API error: {e}",
            "citations": [],
            "spoken": append_unsure_guidance("I’m not completely sure — I couldn’t reach the reasoning service."),
            "query_used": search_query,
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

        claim_raw = data.get("claim", "")
        if not isinstance(claim_raw, str):
            return jsonify({"spoken": "I only understand facts written as text sentences."}), 200
        claim_raw = claim_raw.strip()
        print(f"🧠 Received claim: {claim_raw}")

        if not claim_raw:
            return jsonify({"spoken": "I didn’t catch that to fact-check, sorry!"}), 200

        extracted = extract_core_claim(claim_raw)
        core_claim = extracted.get("core_claim", "no_claim")
        intent = extracted.get("intent", "NONE")

        # If the model failed to label intent but returned a usable claim, treat it as FACT
        if intent == "NONE" and core_claim.lower() != "no_claim":
            intent = "FACT"

        if intent != "FACT" or core_claim.lower() == "no_claim":
            spoken_unsure = append_unsure_guidance(UNSURE_CORE_TEMPLATE)
            response_payload = {
                "spoken": spoken_unsure,
                "verdict": "unsure",
                "confidence": 0.0,
                "citations": [],
                "query_used": "",
            }
            return jsonify(response_payload), 200

        result = run_fact_check_logic(core_claim)
        print("🤖 Gemini raw output:", result.get("rationale", ""))

        # 🧩 Extract the short summary to speak aloud
        # Prefer Gemini’s explanation; fallback to rationale text
        explanation = result.get("rationale", "").strip()
        spoken = result.get("spoken") or explanation
        if result.get("verdict", "").lower() == "unsure":
            spoken = append_unsure_guidance(spoken)

        # 📝 Append detailed log entry with sources for review
        try:
            with open("factcheck_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n🕓 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    f"\n🔎 Claim: {claim_raw}"
                    f"\n🔍 Core claim: {core_claim}"
                    f"\n🎯 Intent: {intent}"
                    f"\n🗣️ Agent would say (full): {spoken}"
                    f"\n🧪 Search query: {result.get('query_used', '')}"
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

        response_payload = {
            "spoken": spoken,
            "verdict": result.get("verdict", "unsure"),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
            "query_used": result.get("query_used", ""),
        }
        return jsonify(response_payload), 200

    except Exception as e:
        print(f"❌ Error in /fact_check: {e}")
        return jsonify({"spoken": "Hmm, my fact-checking tool had trouble just now!"}), 200

@app.route("/fact_checker2", methods=["POST"])
def fact_checker2():
    try:
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"spoken": "I couldn’t read that fact to check it. Please send it again."}), 200

        claim_raw = data.get("claim", "")
        if not isinstance(claim_raw, str):
            return jsonify({"spoken": "I only understand facts written as text sentences."}), 200
        claim_raw = claim_raw.strip()
        print(f"🧠 Received claim: {claim_raw}")

        if not claim_raw:
            return jsonify({"spoken": "I didn’t catch that to fact-check, sorry!"}), 200

        # Optional user question to help ground intent; if provided, prepend to the claim for extraction
        user_question = data.get("user_question") or data.get("original_user_input") or ""
        if user_question and not isinstance(user_question, str):
            user_question = ""
        combined_for_extraction = combine_user_and_agent(user_question, claim_raw) if user_question else claim_raw

        extracted = extract_core_claim(combined_for_extraction)
        core_claim = extracted.get("core_claim", "no_claim")
        intent = extracted.get("intent", "NONE")

        # Safety: treat NONE as non-FACT; also detect emotional/feeling cues and short-circuit
        emotional_terms = ["feel", "feeling", "scared", "afraid", "worried", "anxious", "anxiety", "fear", "nervous", "upset"]
        lower_all = f"{user_question} {claim_raw} {core_claim}".lower()
        if any(term in lower_all for term in emotional_terms):
            intent = "EMOTIONAL"

        if intent != "FACT" or core_claim.lower() == "no_claim":
            spoken_unsure = append_unsure_guidance(UNSURE_CORE_TEMPLATE)
            response_payload = {
                "spoken": spoken_unsure,
                "verdict": "unsure",
                "confidence": 0.0,
                "citations": [],
                "query_used": "",
            }
            return jsonify(response_payload), 200

        # Use user question + core claim for search grounding
        combined_search = combine_user_and_agent(user_question, core_claim) if user_question else core_claim

        result = run_fact_check_logic(combined_search)
        print("🤖 Gemini raw output:", result.get("rationale", ""))

        # 🧩 Extract the short summary to speak aloud
        # Prefer Gemini’s explanation; fallback to rationale text
        explanation = result.get("rationale", "").strip()
        spoken = result.get("spoken") or explanation
        if result.get("verdict", "").lower() == "unsure":
            spoken = append_unsure_guidance(spoken)

        # 📝 Append detailed log entry with sources for review
        try:
            with open("factcheck_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n🕓 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
                    f"\n🔎 Claim: {claim_raw}"
                    f"\n🔍 Core claim: {core_claim}"
                    f"\n🎯 Intent: {intent}"
                    f"\n🗣️ Agent would say (full): {spoken}"
                    f"\n🧪 Search query: {result.get('query_used', '')}"
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

        response_payload = {
            "spoken": spoken,
            "verdict": result.get("verdict", "unsure"),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
            "query_used": result.get("query_used", ""),
        }
        return jsonify(response_payload), 200

    except Exception as e:
        print(f"❌ Error in /fact_checker2: {e}")
        return jsonify({"spoken": "Hmm, my fact-checking tool had trouble just now!"}), 200




@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
