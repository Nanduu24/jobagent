# JobAgent — autonomous job discovery, tailoring, and application tracker

## What this is
A personal job-application pipeline for a new-grad MS CS (AI/ML) on OPT.
Ingests postings from official ATS board APIs, filters by visa sponsorship,
scores match, tailors a resume from a verified fact bank, pre-fills the
application form for human review, and tracks status from email.

## Non-negotiable constraints
1. NEVER auto-submit an application. The agent pre-fills; a human clicks Submit.
   No exceptions, no "autopilot mode" flag.
2. NEVER scrape LinkedIn or Indeed. Use official board APIs only
   (Greenhouse, Lever, Ashby, Workable, SmartRecruiters, Adzuna, USAJobs).
   Scraping = account ban = catastrophic while on OPT.
3. NEVER let the LLM invent a resume claim. Every generated bullet must trace to
   an id in fact_bank.json and must not introduce numbers, tools, titles,
   timeframes, or scale absent from the source fact. A verifier gate enforces
   this; failing bullets are regenerated or dropped, never shipped.
4. "ATS score" is a keyword-overlap heuristic, not a real ATS verdict.
   Name it `match_score`. Do not market it as an ATS pass/fail.
5. Secrets in .env only. Never commit, never log, never print API keys.

## Stack
- Python 3.11, uv for deps
- FastAPI (API) + Postgres 16 (asyncpg + SQLAlchemy 2.0 async)
- LangGraph for the tailoring agent
- Playwright (headed, human-in-loop) for form pre-fill
- Next.js + Tailwind + shadcn/ui for the review queue
- Groq (llama-3.3-70b) as default LLM; provider abstraction so Claude/Gemini swap in
- pytest + respx for HTTP mocking

## Layout
jobagent/
  ingest/      # one adapter per board API, normalize -> Job
  filters/     # sponsorship, hard requirements, dedupe
  scoring/     # embedding sim + LLM rubric -> match_score + breakdown
  tailor/      # LangGraph: select -> rewrite -> verify -> render
  apply/       # Playwright form pre-fill, review queue backend
  track/       # Gmail poll + status classifier
  db/          # models, migrations (alembic)
  llm/         # provider abstraction
web/           # Next.js review queue
data/
  fact_bank.json
  companies.yaml   # board tokens to poll

## Style
- Type hints everywhere, mypy strict on jobagent/
- Pydantic v2 models at every boundary
- Structured logging (structlog), no print()
- Small functions, no god-classes
- Write the test with the code, not after

## Workflow
Work in phases. Do not start a phase until I say so. At the end of each phase:
run tests, run mypy, show me a diff summary, then stop and wait.
## Database setup (reality note)
Postgres 16 is the target. `docker-compose.yml` (image: postgres:16) is the
intended path, but the pipeline was actually developed/verified against a local
**Homebrew** Postgres 16 on a machine with no Docker runtime. If Docker is
unavailable, use `scripts/pg_local.sh start` (port 5433, role/db `jobagent`) —
it matches the default `DATABASE_URL`. Alembic rewrites the driver
(asyncpg -> psycopg) itself; there is a single `DATABASE_URL` to configure.
