# Mentor matching prototype (mBolden)

## Status
**Prototype / demo-grade.** Working end to end, but a few pieces are stand-ins
for production components -- see "Known limitations" below before treating
any part of this as final.

## What this is
An AI-driven mentor-matching engine: bulk resume ingest (real PDF/DOCX files)
-> sentence-transformer embeddings -> similarity-based retrieval -> capacity-aware
assignment -> a results table, all driven from a single page.

The workflow matches how the team actually gets resumes: three buckets --
mentor resumes, mentee resumes, and people open to either -- uploaded
separately, plus one field for "how many mentees can a single mentor take."
Hit one button and get back a sorted, matched list with names and emails for
both sides of every match.

## Structure
- `app/database.py`      - SQLite schema (`people`, `matches`)
- `app/embeddings.py`    - embedding layer (real sentence-transformer, with a
                            local fallback encoder for offline/sandboxed environments)
- `app/resume_parser.py` - PDF/DOCX/TXT text extraction + best-effort name/title/email guessing
- `app/matching.py`      - retrieval (shortlist generation) + capacity-aware assignment
- `app/main.py`          - FastAPI endpoints, serves `frontend/` at `/`
- `frontend/index.html`  - the matching page (upload buckets, capacity field, results table)
- `sample_data/resumes/{mentors,mentees,both}/` - real sample PDF/DOCX files, one per bucket
- `sample_data/resumes.json` - 12 pre-baked sample people, for the quick-demo `seed.py` path
- `seed.py`              - loads `resumes.json`, embeds, writes to DB (fast demo path)
- `ingest_resumes.py`    - CLI equivalent of the bulk upload flow, reading from local folders

## Run it
```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```
Then open **http://127.0.0.1:8000/** in a browser. That's the whole tool:

1. Upload resumes into the mentor / mentee / open-to-both fields (PDF, DOCX, or TXT)
2. Set "mentees per mentor"
3. Click **Run matching**
4. Get a table sorted by match score, with each person's name, internal ID,
   email, and a one-line rationale for the match

Want sample data instead of your own files? `sample_data/resumes/mentors/`,
`.../mentees/`, and `.../both/` each have one real PDF or DOCX you can upload
straight into the page to see it work.

## API (used by the page, callable directly too)
```
POST   /api/bulk-upload-resumes   pool=mentor|mentee|both, files[]  -- ingest one bucket
DELETE /api/reset-all                                               -- clear all people + matches
POST   /api/rebuild-matches                                         -- recompute shortlists (retrieval step)
POST   /api/confirm-assignments?capacity=n                          -- capacity-aware assignment, n applied to every mentor
GET    /api/all-results                                             -- every confirmed match, sorted by score desc
GET    /api/mentees                                                 -- everyone eligible for a shortlist
GET    /api/matches/{mentee_id}                                     -- one mentee's ranked shortlist
```

## Design decisions baked in (per product discussion)
- Which upload bucket a resume goes into IS its role (mentor / mentee / both) --
  no attempt to classify role intent from resume text, since a wrong guess
  there would misdirect a match. Name/title/email guesses are lower-stakes
  (cosmetic if wrong) so those are extracted automatically.
- Mentor capacity is a single global number set at match time (the "mentees
  per mentor" field), not something configured per person.
- People marked "open to both" stay in both pools simultaneously -- can
  appear as a mentor candidate on someone else's shortlist while also having
  their own shortlist as a mentee.
- Each mentee gets a ranked shortlist internally (up to 5, by similarity);
  the results table shows confirmed assignments after the capacity-aware
  pass, sorted by score.

## Known limitations
- **Embeddings**: real `all-MiniLM-L6-v2` sentence-transformer embeddings are
  used when the model can download (needs huggingface.co access). If that's
  blocked (e.g. a locked-down sandbox), `app/embeddings.py` falls back to a
  deterministic hashed bag-of-words encoder so the pipeline still runs --
  but that fallback is *not* semantically meaningful, only useful for
  demoing the plumbing. Check server logs for a fallback warning; if you see
  it in a real deployment, fix network access before trusting match scores.
- **Rationale generation** is extractive (keyword overlap between resumes),
  not LLM-generated. `app/matching.py: generate_rationale()` is a single
  swappable function if richer generated explanations are wanted later.
- **Name/title/department**: name and title are guessed from the first two
  lines of each resume (reliable for standard formats, not for resumes with
  a logo/header image before the text). Department and seniority aren't
  guessed at all and stay blank.
- **Email** is extracted via regex, which is reliable when a resume includes
  one in plain text, but won't catch emails embedded only in an image.
- **No OCR**: scanned/image-only PDFs won't yield any extractable text.
