# Catinci Fact-Check API

Local/Render-friendly fact-checking service used by an ElevenLabs agent to verify answers, rank sources, and speak a kid-safe verdict.

## Branches
- `main`: Stable baseline. Accepts `claim` and returns a verdict/spoken response. No core-claim extraction or intent gating.
- `feat/core-claim-extractor`: Adds an LLM-based core-claim extractor and intent gating (FACT vs EMOTIONAL/SENSITIVE). Accepts `user_question` and `claim`, short-circuits emotional inputs, and uses the extracted core claim for search. This is the branch to use for the new ElevenLabs tool (`fact_check_intent` → `/fact_checker2`).

## Endpoints
- `POST /fact_check` (main flow)
  - Body: `{"claim": "<agent’s answer or user-provided statement>"}`.
  - Returns JSON with `spoken`, `verdict`, `confidence`, `citations`, `query_used`.
- `POST /fact_checker2` (core-claim/intent flow on `feat/core-claim-extractor`)
  - Body: `{"user_question": "<original user ask>", "claim": "<agent’s answer>"}`. `user_question` is optional; `claim` is required.
  - Runs core-claim extraction, intent gating, and uses the core claim for search. Emotional topics may return unsure without searching.
- `GET /health` → `{ "status": "ok" }`.

## Environment
Create a `.env` (or set env vars) with:
- `GOOGLE_API_KEY`
- `GOOGLE_CX_ID` (Custom Search Engine ID)
- `GEMINI_API_KEY`

## Install & Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py  # listens on PORT or 5001
```

## Example Requests
`/fact_check` (baseline):
```bash
curl -X POST http://127.0.0.1:5001/fact_check \
  -H "Content-Type: application/json" \
  -d '{"claim":"Penguins live in the Arctic."}'
```

`/fact_checker2` (core-claim/intent, use this for the new tool):
```bash
curl -X POST http://127.0.0.1:5001/fact_checker2 \
  -H "Content-Type: application/json" \
  -d '{"user_question":"Are bananas radioactive?","claim":"Bananas have a tiny bit of natural radioactivity from potassium, but they are safe to eat."}'
```

## Behavior Highlights
- Domain scoring prefers `.edu`, `.gov`, and trusted science/education sites; penalizes social/forums/low-cred domains.
- Sensitive/emotional guard (on `feat/core-claim-extractor`) avoids searching unsafe topics and returns a gentle unsure message instead.
- Spoken output is preformatted for TTS; consuming agents should read the `spoken` field verbatim.
- Logging: results append to `factcheck_log.txt` with claim, core claim (on `/fact_checker2`), intent, query, and sources.

## ElevenLabs Integration Notes
- Old tool: POSTs to `/fact_check` with `{"claim": ...}`.
- New tool (`fact_check_intent`): POSTs to `/fact_checker2` with `{"user_question": ..., "claim": ...}`. This enables better intent detection and safer searches for kid-facing prompts.

## Deploying to Render (typical settings)
- Build: `pip install -r requirements.txt`
- Start: `python server.py`
- Env vars: `GOOGLE_API_KEY`, `GOOGLE_CX_ID`, `GEMINI_API_KEY`
- Point ElevenLabs to `https://<your-service>.onrender.com/fact_checker2` for the new tool.
