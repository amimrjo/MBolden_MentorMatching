# Mentor matching prototype (mBolden)

## What this is
A working prototype of the AI-driven mentor-matching engine: bulk resume ingest ->
sentence-transformer embeddings -> similarity-based retrieval -> capacity-aware
assignment -> ranked shortlist, shaped to drop directly into the mentee-facing
match results UI.

## Structure
- `app/database.py`   - SQLite schema (people, matches)
- `app/embeddings.py` - embedding layer (real sentence-transformer, with a local
                         fallback encoder for offline/sandboxed environments)
- `app/matching.py`   - retrieval (shortlist generation) + capacity-aware assignment
- `app/main.py`       - FastAPI endpoints
- `sample_data/resumes.json` - 12 sample people across mentor / mentee / both pools
- `seed.py`           - loads sample resumes, computes embeddings, writes to DB

## Run it
```
pip install -r requirements.txt
python seed.py                      # ingest + embed sample resumes
python -m uvicorn app.main:app --reload --port 8000
```
Then:
```
POST /api/rebuild-matches       # (re)compute shortlists for every mentee
POST /api/confirm-assignments   # lock in capacity-aware assignments
GET  /api/mentees               # everyone eligible to request a match
GET  /api/matches/{mentee_id}   # shortlist for one mentee -> UI card shape
```

## Design decisions baked in (per product discussion)
- Each mentee gets a ranked shortlist of up to 5 mentors, not a single match.
- A mentor can be assigned up to `mentor_capacity` mentees (3-5, set per mentor).
- People marked "open to both" stay in both pools simultaneously and can appear
  as a mentor candidate on one shortlist while having their own shortlist as a mentee.
- Shortlist generation (what mentees see) is separate from confirmed assignment
  (what capacity allows) -- see `app/matching.py` docstring for why.

## Known limitation in this environment
This sandbox's network allowlist doesn't include huggingface.co, so the real
`all-MiniLM-L6-v2` sentence-transformer weights can't be downloaded here.
`app/embeddings.py` falls back to a deterministic hashed bag-of-words encoder
so the full pipeline still runs end-to-end. In a real deployment (or any
environment with HF access) the real model loads automatically and nothing
else in the codebase needs to change.

## Real resume ingest (PDF/DOCX)
Ingest is organized around three buckets -- mentor resumes, mentee resumes,
and open-to-both resumes -- matching a website form with three separate
upload fields. Which bucket a file goes into IS its role_pool: no per-file
classification of "is this person a mentor or mentee" is needed, which
sidesteps having to reliably infer that from free-text prose.

**Bulk (three folders, one per bucket)**
```
python ingest_resumes.py \
    --mentor-dir sample_data/resumes/mentors \
    --mentee-dir sample_data/resumes/mentees \
    --both-dir   sample_data/resumes/both \
    --default-mentor-capacity 5
```

**Bulk via API (one call per bucket, matches a 3-field upload form)**
```
POST /api/bulk-upload-resumes
     multipart form: pool=mentor|mentee|both, default_mentor_capacity, files[]
```

Name and title are best-effort guessed from the top of each resume
(`app/resume_parser.guess_name_and_title` -- first line is usually a name,
second line is usually a title). Department/seniority aren't guessed and
stay blank; a wrong name/title guess is a cheap cosmetic fix, unlike a wrong
role classification which would misdirect a match.

Call `/api/rebuild-matches` after a batch of uploads finishes to refresh shortlists.

Supported file types: .pdf, .docx, .txt. Scanned/image-only PDFs won't extract
text -- that would need an OCR step (not included).

Sample real files included under `sample_data/resumes/{mentors,mentees,both}/`.
