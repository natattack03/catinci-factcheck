"""
client_tool.py – Catinci Fact-Check Client Tool (Simulation Mode)
-----------------------------------------------------------------
Simulates ElevenLabs Agent behavior locally. 
Sends claims to the local Flask fact-check API and produces spoken-style responses.
Logs full results with timestamps and sources.
"""

import os
from urllib.parse import urlparse
import time
import re
import json
import textwrap
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# -------------------------------------------------------------------
# 1. Load environment variables
# -------------------------------------------------------------------
load_dotenv()
FACTCHECK_API = os.getenv("FACTCHECK_API_URL", "http://127.0.0.1:5001/fact_check")


# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------

def clean_rationale(text: str) -> str:
    """Remove Markdown fences and embedded JSON from Gemini output."""
    if not text:
        return ""

    # Remove Markdown code blocks and backticks
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "").strip()

    # Try to parse JSON if it exists inside the text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "rationale" in obj:
            text = obj["rationale"]
    except Exception:
        pass  # leave text as-is if it's not valid JSON

    # Remove leftover braces or quotes
    text = re.sub(r"[{}\"\']", "", text)
    return text.strip()


 

def simplify_language(text: str) -> str:
    """Swap out heavier vocabulary for kid-friendly wording."""
    replacements = [
        (r"\boverwhelmingly contradicts\b", "really doesn't match"),
        (r"\boverwhelming contradicts\b", "really doesn't match"),
        (r"\bcontradicts\b", "goes against"),
        (r"\bcontradicted\b", "went against"),
        (r"\brefutes\b", "shows it's not right"),
        (r"\brefuted\b", "showed it wasn't right"),
        (r"\bdebunks\b", "shows it's wrong"),
        (r"\bdebunked\b", "showed it was wrong"),
        (r"\bmisinformation\b", "wrong idea"),
        (r"\bdisinformation\b", "false story"),
        (r"\bcorroborates\b", "matches"),
        (r"\bcorroborated\b", "matched"),
        (r"\bsubstantiates\b", "backs up"),
        (r"\bsubstantiated\b", "backed up"),
        (r"\bdisputes\b", "argues against"),
        (r"\bdisputed\b", "was argued about"),
        (r"\bunequivocally\b", "very clearly"),
        (r"\bunequivocal\b", "very clear"),
        (r"\breputable\b", "trusted"),
        (r"\bclarify\b", "explain"),
        (r"\bclarifies\b", "explains"),
        (r"\bclarified\b", "explained"),
        (r"\bsnippets?\b", "notes"),
        (r"\boverwhelmingly\b", "really"),
        (r"\bprimarily\b", "mostly"),
        (r"\bapproximately\b", "about"),
        (r"\bsignificant\b", "big"),
        (r"\borbits\b", "goes around"),
        (r"\borbit\b", "go around"),
        (r"\borbited\b", "went around"),
        (r"\brevolves\b", "goes around"),
        (r"\brevolve\b", "go around"),
        (r"\brevolved\b", "went around"),
        (r"\borbits around\b", "goes around"),
        (r"\borbit around\b", "go around"),
        (r"\butilize\b", "use"),
        (r"\bapproximately\b", "about"),
        (r"\bdont\b", "don't"),
        (r"\bcant\b", "can't"),
        (r"\bbarycenter\b", "center of the solar system"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Remove citation-style numeric brackets (e.g., [1], [2])
    text = re.sub(r"\[\d+\]", "", text)
    # Collapse extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Fix duplicate 'around around'
    text = re.sub(r"\b(go|goes) around around\b", r"\1 around", text, flags=re.IGNORECASE)
    # Fix awkward passive constructions after replacements
    text = re.sub(r"\b(went|goes)\s+against by\b", "is not correct", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdirectly\s+went\s+against\b", "clearly goes against", text, flags=re.IGNORECASE)
    return text

def polish_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().strip('"\' ,;:')
    if not text:
        return ""
    text = re.sub(r"^(?:although|though|while)[,\s]*[^,]*,\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return ""
    # Capitalize first letter
    text = text[0].upper() + text[1:]
    # Fix spacing before periods and duplicate punctuation
    text = re.sub(r"\s+\.(?=\W|$)", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text

def is_complete_simple_sentence(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t.split()) < 6:
        return False
    bad_endings = [
        "because", "such as", "for example", "for instance", "like", "this is what causes",
        "more than", "at least", "including", "include", "includes"
    ]
    low = t.lower()
    trimmed = low.rstrip(" .")
    if any(trimmed.endswith(be) for be in bad_endings):
        return False
    # Require a simple verb indicating a factual claim
    if not re.search(r"\b(is|are|has|have|can|cannot|can't|includes?|contain|contains|means|refers|provide|provides|allow|allows|recognized|recognised|remain|remains)\b", low):
        return False
    if '?' in t:
        return False
    if "you've probably heard" in low or low.startswith("have you ever heard") or "common phrase" in low:
        return False
    return True


VERDICT_PREFIXES = {
    "true": "That’s true—",
    "false": "That’s not true—",
    "unsure": "I’m not completely sure—"
}

# Prefer simple, trusted names for known domains
TRUSTED_DOMAINS = {
    "wikipedia.org": "Wikipedia",
    "nasa.gov": "NASA",
    "britannica.com": "Britannica",
    "nationalgeographic.com": "National Geographic",
    "kids.nationalgeographic.com": "National Geographic Kids",
    "smithsonianmag.com": "Smithsonian",
    "noaa.gov": "NOAA",
    "nih.gov": "NIH",
    "cdc.gov": "CDC",
    "who.int": "WHO",
    "nature.com": "Nature",
}

LOW_QUALITY_DOMAINS = [
    "reddit.com", "stackexchange.com", "quora.com", "facebook.com", "twitter.com", "x.com", "medium.com",
    "wordreference.com", "youtube.com", "youtu.be", "tiktok.com", "instagram.com"
]

# Easy-to-read reference sites for general facts (tie-breakers)
READABLE_DOMAINS = {
    "wikipedia.org": 35,
    "britannica.com": 35,
    "nationalgeographic.com": 35,
    "kids.nationalgeographic.com": 45,
    "smithsonianmag.com": 30,
    "science.org": 20,
}


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def registrable_domain(netloc: str) -> str:
    if not netloc:
        return ""
    host = netloc.split(':', 1)[0]
    parts = host.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return host

def root_brand_from_domain(domain: str) -> str:
    if not domain:
        return ""
    parts = domain.split('.')
    if len(parts) >= 2:
        core = parts[-2]
        return core.capitalize()
    return domain

def citation_score(c: dict) -> int:
    url = c.get("url", "")
    title = (c.get("title") or "").lower()
    d = domain_from_url(url)
    score = 0
    for dom in TRUSTED_DOMAINS:
        if d.endswith(dom):
            score += 100
            break
    if any(d.endswith(suf) for suf in [".gov", ".edu"]):
        score += 80
    if d.endswith(".org"):
        score += 40
    if any(bad in d for bad in LOW_QUALITY_DOMAINS):
        score -= 120
    if any(w in title for w in ["ask", "forum", "discussion", "thread"]):
        score -= 20
    # Readability tie-breaker
    for dom, bonus in READABLE_DOMAINS.items():
        if d.endswith(dom):
            score += bonus
            break
    # Extra penalty for video/social platforms regardless of title
    if any(d.endswith(v) for v in ["youtube.com", "youtu.be", "tiktok.com", "instagram.com"]):
        score -= 100
    # Penalize clickbaity titles and recipe/how-to patterns
    if re.search(r"\bhow\s+to\b", title) or "!" in title or "?" in title:
        score -= 20
    if re.search(r"\b(recipe|homemade|step[- ]?by[- ]?step|steps|guide|ultimate|easy|delicious|best|make|making|tutorial)\b", title):
        score -= 35
    return score

def clean_source_display(c: dict) -> str:
    url = c.get("url", "")
    title = (c.get("title") or "").strip()
    d = domain_from_url(url)
    rd = registrable_domain(d)
    if rd:
        return rd
    # Fall back to a cleaned-up title or the raw domain
    if title:
        title = title.split('|')[0]
        title = title.split(' — ')[0]
        title = title.split(' - ')[0]
        title = title.split(':')[0]
        title = re.sub(r"\br/\w+\b", "", title, flags=re.IGNORECASE).strip()
        if len(title.split()) <= 4 and len(title) >= 3:
            return title
    return (d or "a trusted website").strip().strip(' .:?—-')

MONTH_WORDS = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
]

def is_clean_fact_sentence(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    low = t.lower()
    if any(m in low for m in MONTH_WORDS) or re.search(r"\b20\d{2}\b", low):
        return False
    if any(x in low for x in ["http://", "https://", "www."]):
        return False
    # Avoid nested source attributions inside sentences; we add our own top-level
    if "according to" in low:
        return False
    if t.endswith(":") or t.endswith("-"):
        return False
    if len(t.split()) < 5:
        return False
    # Avoid discourse/meta/connective openers and inference wording
    if low.startswith("so ") or low.startswith("thus ") or low.startswith("therefore ") or low.startswith("in conclusion") or low.startswith("overall "):
        return False
    # Avoid conjunction-led sentences (often fragments from quotes)
    if low.startswith("and ") or low.startswith("but ") or low.startswith("or "):
        return False
    # Avoid rhetorical contrast openers that often precede incomplete context
    if low.startswith("yet ") or low.startswith("however "):
        return False
    # Avoid deictic/pronoun-led generic openers
    if low.startswith("this ") or low.startswith("that "):
        return False
    if " infer " in f" {low} ":
        return False
    # Drop slang/enum/parenthetical noise
    if " til " in f" {low} " or "today i learned" in low:
        return False
    if "#" in s:
        return False
    # Avoid video/social content and promotional chatter
    bad_media_terms = [
        "youtube", "video", "subscribe", "channel", "click", "watch", "tiktok", "instagram",
        "mrbeast", "hershey", "feastables"
    ]
    if any(term in low for term in bad_media_terms):
        return False
    if re.search(r"\bhow\s+to\b", low):
        return False
    # Allow parentheses (units or clarifications) as long as the sentence is otherwise clean
    # Guard against incomplete "because ... never" patterns
    if "because there are never" in low or re.search(r"\bbecause\b.*\bnever\b\.?$", low):
        return False
    if "you've probably heard" in low or low.startswith("have you ever heard") or "common phrase" in low:
        return False
    # Allow research-y sentences (e.g., "Does not show that", "We demonstrate...")
    if "sources like" in low or "sources" in low:
        return False
    if re.search(r"\bnotes?\b", low):
        return False
    if "expedition" in low or "expeditions" in low:
        return False
    if "forum" in low or "forums" in low or "wordreference" in low:
        return False
    if low.count(',') > 4:
        return False
    if low.count(' and ') > 3:
        return False
    words = t.split()
    if len(words) > 40:
        return False
    if ("no penguins" in low or "no penguin" in low) and "south pole" in low:
        return False
    # Prefer simple present/clear facts
    return any(w in low for w in [" is ", " are ", " live ", " lives ", " go ", " goes ", " grow ", " grows ", " not "])

DEFAULT_PRIMARY = {
    "true": "",
    "false": "",
    "unsure": ""
}

DEFAULT_EXTRA = {
    "true": "",
    "false": "This is a common mistake people sometimes believe.",
    "unsure": "We need a bit more clear information to be sure."
}


def keywords_from_query(query: Optional[str]) -> set:
    if not query:
        return set()
    q = re.sub(r"[^a-zA-Z\s]", " ", query.lower())
    words = [w for w in q.split() if len(w) >= 4]
    stop = {"that","this","there","which","about","because","with","without","have","has","from","into","after","before","between","among","their","these","those","been","very","more","most","some","such","only","over","under","into","your","ours","they","them","what","were","will","would","could","should","might"}
    return {w for w in words if w not in stop}


def format_spoken_response(
    verdict: str,
    rationale: str,
    max_chars: Optional[int] = None,
    citations=None,
    query: Optional[str] = None
) -> str:
    """
    Creates a conversational spoken summary with one source and short explanation.
    Example:
    "That’s not true — According to Aurora Expeditions, penguins live only in the Southern Hemisphere, mainly around Antarctica."
    """
    verdict = (verdict or "unsure").lower()

    prefix = VERDICT_PREFIXES.get(verdict, VERDICT_PREFIXES["unsure"])

    # Choose best citation by domain quality and quote quality
    source_name = ""
    best_citations = []
    if citations:
        # Score citations by domain + presence of clean quote sentences + overlap with query keywords
        qkeys = keywords_from_query(query)
        scored = []
        for c in citations:
            base = citation_score(c)
            quote = (c.get("quote") or "").strip()
            clean_count = 0
            overlap_bonus = 0
            numbers_bonus = 0
            if quote:
                qparts = re.split(r'(?<=[.!?]) +', simplify_language(re.sub(r"\s+", " ", quote)))
                for qp in qparts:
                    if is_clean_fact_sentence(qp):
                        clean_count += 1
                        if qkeys and any(k in qp.lower() for k in qkeys):
                            overlap_bonus += 15
                        if re.search(r"\d", qp) or re.search(r"\b(million|billion|thousand|km|kilomet|square|sq\s*mi|percent|%)\b", qp, flags=re.IGNORECASE):
                            numbers_bonus += 10
                        if clean_count >= 2:
                            break
        # boost good quotes, penalize none
            score = base + (clean_count * 25) + overlap_bonus + numbers_bonus - (0 if clean_count else 90)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_citations = [c for _, c in scored]
        if best_citations:
            source_name = clean_source_display(best_citations[0])

    # Clean the rationale and extract first 1–2 natural sentences
    rationale = rationale.strip()
    rationale = re.sub(r"(?i)snippets?\s*\d.*", "", rationale)
    rationale = re.sub(r"(?i)evidence\s*\d.*", "", rationale)
    rationale = re.sub(r"(?i)multiple sources.*", "", rationale)
    rationale = re.sub(r"(?i)multiple reputable sources.*", "", rationale)
    sentences = re.split(r'(?<=[.!?]) +', rationale)
    sentences = [s.strip() for s in sentences if s.strip()]
    short_rationale = " ".join(sentences[:2]).strip() if sentences else rationale.strip()

    # Tone cleanup (more natural word choices)
    short_rationale = short_rationale.replace("All provided evidence", "Research shows")
    short_rationale = short_rationale.replace("unequivocally states", "shows clearly")
    short_rationale = short_rationale.replace("explicitly state", "explain")
    short_rationale = short_rationale.replace("the presence of", "because of")
    short_rationale = short_rationale.replace("factual one", "scientific fact")
    short_rationale = re.sub(r"Research shows\s+explain[s]?", "Research shows", short_rationale, flags=re.IGNORECASE)
    short_rationale = simplify_language(short_rationale)

    # Break into primary and secondary sentences for format consistency
    rationale_sentences = re.split(r'(?<=[.!?]) +', short_rationale)
    rationale_sentences = [simplify_language(s.strip()) for s in rationale_sentences if s.strip()]

    quote_sentences = []
    if best_citations:
        for c in best_citations:
            quote = (c.get("quote") or "").strip()
            if not quote:
                continue
            quote = re.sub(r"\s+", " ", quote)
            quote = simplify_language(quote)
            quote_parts = re.split(r'(?<=[.!?]) +', quote)
            for part in quote_parts:
                part_clean = part.strip()
                if not part_clean:
                    continue
                if is_clean_fact_sentence(part_clean) and is_complete_simple_sentence(part_clean):
                    quote_sentences.append(part_clean.rstrip(".!?"))
                if len(quote_sentences) >= 2:
                    break
            if len(quote_sentences) >= 2:
                break

    def pick_relation_sentence(sent_list, qkeys):
        relation_phrases = [
            "comes from", "come from", "made from", "is made from", "are made from",
            "derived from", "is derived from", "are derived from", "grown from",
            "grow on", "grows on", "produced from", "is produced from", "are produced from"
        ]
        best = ""
        for s in sent_list:
            low = s.lower()
            if any(r in low for r in relation_phrases):
                if not qkeys or sum(1 for k in qkeys if k in low) >= 1:
                    best = s
                    break
        return best

    filtered_sentences = []
    for sentence in rationale_sentences:
        lowered = sentence.lower()
        if lowered.startswith("the claim states") or "the claim that" in lowered or lowered.startswith("the claim "):
            continue
        if "multiple reputable sources" in lowered:
            continue
        if "sources like" in lowered or "sources" in lowered:
            continue
        if "according to" in lowered:
            continue
        if re.search(r"\bnotes?\b", lowered):
            continue
        if "you've probably heard" in lowered or "common phrase" in lowered:
            continue
        if "multiple notes" in lowered or "notes explain" in lowered:
            continue
        if "notes clarify" in lowered or "snippet" in lowered:
            continue
        if "research shows" in lowered or "consistently state" in lowered or "consistently states" in lowered:
            continue
        if "forum" in lowered or "forums" in lowered or "wordreference" in lowered:
            continue
        # Skip overly compound rationale pieces with multiple topics
        if lowered.count(',') > 4:
            continue
        if verdict == "true" and any(word in lowered for word in ["contradict", "incorrect", "false", "not true", "wrong"]):
            continue
        if verdict == "false" and any(word in lowered for word in ["support", "correct", "true", "back up", "accurate"]):
            continue
        if is_complete_simple_sentence(sentence):
            filtered_sentences.append(sentence)

    # Prefer clean quote sentences first, then filtered rationale sentences
    fact_candidates = quote_sentences[:] if quote_sentences else []
    lower_seen = {sentence.lower() for sentence in fact_candidates}
    for qs in filtered_sentences:
        qs_lower = qs.lower()
        if qs_lower not in lower_seen:
            fact_candidates.append(qs)
            lower_seen.add(qs_lower)

    # If still empty, leave as-is; defaults will supply a simple explanation

    def pick_fact_sentence(candidates):
        relation_phrases = [
            "comes from", "come from", "made from", "is made from", "are made from",
            "derived from", "is derived from", "are derived from", "grown from",
            "grow on", "grows on", "are called", "is called"
        ]
        def sentence_score(sentence: str) -> float:
            lowered = sentence.lower()
            score = 0.0
            informative_words = [
                "because", "since", "so", "means", "found", "finds", "found that",
                "shows that", "shows how", "live", "lives", "grows", "grow",
                "has", "have", "is", "are", "includes", "contains", "explains", "happens"
            ]
            if lowered.startswith("research shows"):
                score += 1.0
            if any(word in lowered for word in informative_words):
                score += 2.5
            if any(phrase in lowered for phrase in relation_phrases):
                score += 4.0
            if "not" in lowered or "n't" in lowered:
                score += 0.5
            if "helpers" in lowered:
                score -= 1.5
            if "claim" in lowered or "idea" in lowered:
                score -= 3.0
            # Penalize pronoun-led sentences (prefer subject nouns like 'elephants', 'Africa')
            if lowered.startswith("they ") or lowered.startswith("we ") or lowered.startswith("you ") or lowered.startswith("i "):
                score -= 1.5
            if len(sentence.split()) >= 6:
                score += 1.0
            # Keyword overlap with query to steer toward on-topic nouns
            if 'query' in locals() and query:
                try:
                    qk = keywords_from_query(query)
                    if qk and any(k in lowered for k in qk):
                        score += 4.0
                    # Extra boost if at least two keywords occur
                    if qk and sum(1 for k in qk if k in lowered) >= 2:
                        score += 2.0
                except Exception:
                    pass
            return score

        best_sentence = candidates[0]
        best_score = sentence_score(best_sentence)
        for sentence in candidates[1:]:
            score = sentence_score(sentence)
            if score > best_score:
                best_sentence = sentence
                best_score = score
        return best_sentence.rstrip(".!?")

    if fact_candidates:
        primary_sentence = pick_fact_sentence(fact_candidates)
        remaining_candidates = [
            c for c in fact_candidates
            if c.lower().rstrip(".!?") != primary_sentence.lower().rstrip(".!?")
        ]
        # If the chosen primary is incomplete, try a complete alternative
        if not is_complete_simple_sentence(primary_sentence):
            for cand in remaining_candidates:
                if is_complete_simple_sentence(cand):
                    primary_sentence = cand
                    # rebuild remaining after swap
                    remaining_candidates = [
                        c for c in fact_candidates
                        if c.lower().rstrip(".!?") != primary_sentence.lower().rstrip(".!?")
                    ]
                    break
        extra_sentence = pick_fact_sentence(remaining_candidates) if remaining_candidates else ""
        # If the chosen primary is meta-like (talks about "the claim" or "research"), prefer a factual alternative
        meta_terms = ["the claim", "research shows", "multiple sources", "evidence"]
        if any(mt in primary_sentence.lower() for mt in meta_terms):
            for cand in remaining_candidates + fact_candidates:
                cl = cand.lower()
                if all(mt not in cl for mt in meta_terms):
                    primary_sentence = cand.rstrip(".!?")
                    # rebuild remaining after swap
                    remaining_candidates = [
                        c for c in fact_candidates
                        if c.lower().rstrip(".!?") != primary_sentence.lower().rstrip(".!?")
                    ]
                    extra_sentence = pick_fact_sentence(remaining_candidates) if remaining_candidates else extra_sentence
                    break
    else:
        # Fallback: use top citation’s quote with a relation-first strategy,
        # then sensible minimal fallback to first two sentences.
        used_quote_fallback = False
        primary_sentence = ""
        extra_sentence = ""
        if best_citations:
            quote = (best_citations[0].get("quote") or "").strip()
            if quote:
                raw_parts = [p.strip() for p in re.split(r'(?<=[.!?]) +', simplify_language(re.sub(r"\s+", " ", quote))) if p.strip()]
                # Remove nested attributions and URLs
                raw_parts = [p for p in raw_parts if "according to" not in p.lower() and "http" not in p.lower() and "www." not in p.lower()]
                # Try relation sentence first
                qkeys_all = keywords_from_query(query) if query else set()
                rel = pick_relation_sentence(raw_parts, qkeys_all)
                parts = []
                if rel:
                    if is_clean_fact_sentence(rel) and is_complete_simple_sentence(rel):
                        parts = [rel]
                        # pick next best complete sentence different from primary
                        for p in raw_parts:
                            if p != rel and is_clean_fact_sentence(p) and is_complete_simple_sentence(p):
                                parts.append(p)
                                break
                else:
                    # minimal filters to avoid process/how-to chatter
                    for p in raw_parts:
                        low = p.lower()
                        if any(m in low for m in MONTH_WORDS) or re.search(r"\b20\d{2}\b", low):
                            continue
                        if re.search(r"\b(recipe|homemade|step[- ]?by[- ]?step|steps|guide|ultimate|easy|delicious|best|make|making|tutorial|process)\b", low):
                            continue
                        if re.search(r"\bhow\s+to\b", low):
                            continue
                        if any(term in low for term in ["youtube", "video", "subscribe", "channel", "click", "watch", "tiktok", "instagram", "mrbeast", "hershey", "feastables"]):
                            continue
                        if is_clean_fact_sentence(p) and is_complete_simple_sentence(p):
                            parts.append(p)
                        if len(parts) >= 2:
                            break
                if parts:
                    used_quote_fallback = True
                    primary_sentence = parts[0].rstrip(".!?")
                    if len(parts) > 1:
                        extra_sentence = parts[1].rstrip(".!?")
        if not primary_sentence:
            primary_sentence = DEFAULT_PRIMARY.get(verdict, DEFAULT_PRIMARY["unsure"]) or ""

    # Final guard: if primary is still empty, try a minimal raw sentence from top citation
    if not primary_sentence and best_citations:
        quote = (best_citations[0].get("quote") or "").strip()
        if quote:
            raw_parts = [p.strip() for p in re.split(r'(?<=[.!?]) +', re.sub(r"\s+", " ", quote)) if p.strip()]
            for p in raw_parts:
                low = p.lower()
                if "according to" in low or "http" in low or "www." in low:
                    continue
                if len(p.split()) < 6:
                    continue
                primary_sentence = p.rstrip(".!?")
                break

    primary_sentence = polish_sentence(simplify_language(primary_sentence).strip())
    if not primary_sentence:
        primary_sentence = DEFAULT_PRIMARY.get(verdict, DEFAULT_PRIMARY["unsure"])

    if not extra_sentence:
        if quote_sentences:
            for qs in quote_sentences:
                if qs.lower() != primary_sentence.lower():
                    extra_sentence = qs
                    break
        if not extra_sentence:
            extra_sentence = DEFAULT_EXTRA.get(verdict, DEFAULT_EXTRA["unsure"])
    # If extra_sentence is present but not a complete simple sentence, try another candidate before falling back
    if extra_sentence and not is_complete_simple_sentence(extra_sentence):
        alt = ""
        for cand in remaining_candidates if 'remaining_candidates' in locals() else []:
            if cand.lower().rstrip(".!?") != primary_sentence.lower().rstrip(".!?") and is_complete_simple_sentence(cand):
                alt = cand
                break
        if alt:
            extra_sentence = alt

    # Avoid repeating the same assertion (e.g., "No, bats are not blind.") twice
    def core(s: str) -> str:
        s = s.strip().lower()
        if s.startswith("no, "):
            s = s[4:]
        return re.sub(r"\s+", " ", s)
    if core(extra_sentence) == core(primary_sentence):
        extra_sentence = DEFAULT_EXTRA.get(verdict, DEFAULT_EXTRA["unsure"]) if verdict in DEFAULT_EXTRA else ""

    extra_sentence = polish_sentence(simplify_language(extra_sentence.rstrip(".!?")))
    # Only enforce completeness if extra sentence did not come straight from quote fallback
    if 'used_quote_fallback' in locals() and used_quote_fallback:
        pass
    else:
        if not is_complete_simple_sentence(extra_sentence):
            extra_sentence = DEFAULT_EXTRA.get(verdict, DEFAULT_EXTRA["unsure"]) or ""
    extra_sentence = extra_sentence.rstrip(".!?") + "."

    # Build source phrase, even if we have to fall back
    if source_name:
        intro = f"According to {source_name}, "
    else:
        intro = "According to what I found, "

    # Combine prefix, source, and rationale
    if extra_sentence and extra_sentence.strip():
        spoken = f"{prefix}{intro}{primary_sentence}. {extra_sentence}"
    else:
        spoken = f"{prefix}{intro}{primary_sentence}."

    # Optional trim for preview
    if max_chars is not None and len(spoken) > max_chars:
        spoken = spoken[:max_chars].rsplit(' ', 1)[0] + "…"

    if not spoken.endswith("."):
        spoken += "."

    return spoken








# -------------------------------------------------------------------
# 3. Fact-check logic + logging
# -------------------------------------------------------------------

def fact_check_tool(input_text: str):
    """Send text to Flask fact-check API and log formatted output."""
    print(f"\n🔍 Fact check requested for: {input_text}")
    

    friendly_issue = ""
    debug_detail = ""

    try:
        r = requests.post(FACTCHECK_API, json={"query": input_text}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        friendly_issue = "The fact-check server took too long to reply."
        debug_detail = "timeout"
        data = {
            "verdict": "unsure",
            "rationale": "I couldn't reach the fact-check server just now to double-check this claim.",
            "citations": []
        }
    except Exception as e:
        friendly_issue = "I couldn't reach the fact-check server right now."
        debug_detail = str(e)
        data = {
            "verdict": "unsure",
            "rationale": "I couldn't reach the fact-check server right now to check this claim.",
            "citations": []
        }

    if friendly_issue:
        print(f"⚠️ {friendly_issue} I’ll share what I can for now.")

    rationale_clean = clean_rationale(data.get("rationale", ""))
    spoken_short = format_spoken_response(
        data.get("verdict", "unsure"),
        rationale_clean,
        max_chars=None,
        citations=data.get("citations", []),
        query=input_text
    )
    spoken_full = format_spoken_response(
        data.get("verdict", "unsure"),
        rationale_clean,
        max_chars=None,
        citations=data.get("citations", []),
        query=input_text
    )

    print("\n🗣️ Agent would say:")
    print(textwrap.fill(spoken_short, width=100))
    print()

    # ✅ Log full version with timestamp and sources
    with open("factcheck_log.txt", "a") as f:
        f.write(
            f"\n🕓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"\n🔎 Claim: {input_text}"
            f"\n🗣️ Agent would say (full): {spoken_full}"
        )

        if friendly_issue:
            f.write(f"\n⚠️ Note: {friendly_issue}")
            if debug_detail:
                f.write(f" (details: {debug_detail})")

        cites = data.get("citations", []) or []
        if cites:
            f.write("\n📚 Sources:")
            for c in cites[:5]:
                f.write(f"\n  • {c.get('title','')} — {c.get('url','')}")
        f.write(f"\n{'-'*80}\n")

    return data


# -------------------------------------------------------------------
# 4. Local simulation loop
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("🧩 ElevenLabs Agent simulation mode.\n")
    print("✅ Fact-check client ready. Type a claim to simulate an ElevenLabs request.\n")

    while True:
        query = input("Enter a claim: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        fact_check_tool(query)
        time.sleep(1)
 
