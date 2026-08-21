"""
FastAPI backend for the mentor-matching prototype.

Endpoints:
  POST   /api/bulk-upload-resumes   -- upload a batch of resumes into one pool (mentor/mentee/both)
  DELETE /api/reset-all             -- clear all people + matches (fresh batch run)
  POST   /api/rebuild-matches       -- recompute shortlists for every mentee (retrieval step)
  POST   /api/confirm-assignments   -- capacity-aware assignment; ?capacity=n applies globally
  GET    /api/mentees               -- list everyone who can request a match
  GET    /api/matches/{mentee_id}   -- shortlist for one mentee, shaped for the UI cards
  GET    /api/all-results           -- every confirmed match across everyone, sorted by score
"""
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import get_conn, init_db
from .matching import confirm_assignments, generate_shortlists
from .resume_parser import UnsupportedFileType, extract_text, guess_email, guess_name_and_title

from concurrent.futures import ThreadPoolExecutor

from .embeddings import embed_batch, embed_text

VALID_ROLE_POOLS = {"mentor", "mentee", "both"}

app = FastAPI(title="Mentor Matching API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


def _initials(name: str) -> str:
    parts = name.split()
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else name[:2].upper()


@app.delete("/api/reset-all")
def reset_all():
    """Clears all people + matches. Called before a fresh batch run from the UI."""
    conn = get_conn()
    conn.execute("DELETE FROM matches")
    conn.execute("DELETE FROM people")
    conn.commit()
    conn.close()
    return {"status": "cleared"}


@app.post("/api/bulk-upload-resumes")
async def bulk_upload_resumes(
    pool: str = Form(...),  # 'mentor' | 'mentee' | 'both'
    files: list[UploadFile] = File(...),
):
    """
    Bulk ingest for the three-bucket upload flow: one call per bucket
    (mentor resumes, mentee resumes, open-to-both resumes). Which bucket a
    file lands in IS its role_pool -- no per-file role classification needed.
    Name/title/email are best-effort extracted from the resume text; email
    uses a regex (reliable), name/title use a first-two-lines heuristic
    (a wrong guess there is a cosmetic issue, not a misdirected match).
    Mentor capacity is NOT set here -- it's applied globally at match time
    via /api/confirm-assignments?capacity=n, matching the single "how many
    mentees per mentor" field on the matching page.
    """
    if pool not in VALID_ROLE_POOLS:
        raise HTTPException(400, f"pool must be one of {VALID_ROLE_POOLS}")

    conn = get_conn()
    results = []

        # step 1: save + extract text for every file IN PARALLEL -- this is where
    # OCR time (the real bottleneck for scanned resumes) actually gets clawed back
    def _save_and_extract(upload):
        suffix = Path(upload.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            text = extract_text(tmp_path)
            return (upload.filename, text, None)
        except UnsupportedFileType as e:
            return (upload.filename, None, str(e))
        finally:
            tmp_path.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool_exec:
        extracted = list(pool_exec.map(_save_and_extract, files))

    valid = [(fname, text) for fname, text, err in extracted if text and text.strip()]
    for fname, text, err in extracted:
        if err:
            results.append({"filename": fname, "status": "skipped", "reason": err})
        elif not text or not text.strip():
            results.append({"filename": fname, "status": "skipped", "reason": "no extractable text"})

    if not valid:
        conn.commit()
        conn.close()
        return {"pool": pool, "ingested": 0, "total": len(files), "details": results}

    # step 2: embed everything in ONE batched call instead of one call per file
    vectors = embed_batch([text for _, text in valid])

    # step 3: DB writes stay sequential (SQLite doesn't like concurrent writers)
    for (fname, text), vector in zip(valid, vectors):
        name, title = guess_name_and_title(text)
        email = guess_email(text)
        cur = conn.execute(
            """INSERT INTO people (name, email, title, department, seniority, role_pool, resume_text, mentor_capacity)
               VALUES (?, ?, ?, '', '', ?, ?, 0)""",
            (name or fname, email, title, pool, text),
        )
        conn.execute(
            "UPDATE people SET embedding = ? WHERE id = ?",
            (json.dumps(vector), cur.lastrowid),
        )
        results.append({
            "filename": fname, "status": "ingested", "id": cur.lastrowid,
            "guessed_name": name, "guessed_title": title, "guessed_email": email,
        })
    conn.commit()
    conn.close()

    ingested = sum(1 for r in results if r["status"] == "ingested")
    return {"pool": pool, "ingested": ingested, "total": len(files), "details": results}


@app.post("/api/rebuild-matches")
def rebuild_matches():
    generate_shortlists()
    return {"status": "shortlists regenerated"}


@app.post("/api/confirm-assignments")
def confirm(capacity: int = 5):
    confirmed, total = confirm_assignments(capacity_override=capacity)
    return {"confirmed": confirmed, "total_candidate_pairs": total, "capacity_used": capacity}


@app.get("/api/mentees")
def list_mentees():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, email, title, department FROM people WHERE role_pool IN ('mentee','both') ORDER BY name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/matches/{mentee_id}")
def get_matches(mentee_id: int):
    conn = get_conn()
    mentee = conn.execute("SELECT * FROM people WHERE id = ?", (mentee_id,)).fetchone()
    if not mentee:
        raise HTTPException(404, "mentee not found")

    rows = conn.execute(
        """SELECT m.rank, m.score, m.rationale, m.confirmed,
                  p.id AS mentor_id, p.name, p.title, p.department
           FROM matches m JOIN people p ON p.id = m.mentor_id
           WHERE m.mentee_id = ?
           ORDER BY m.rank ASC""",
        (mentee_id,),
    ).fetchall()
    conn.close()

    # shaped to drop straight into the mentor match results UI cards
    cards = [
        {
            "rank": r["rank"],
            "mentor_id": r["mentor_id"],
            "name": r["name"],
            "initials": _initials(r["name"]),
            "title": r["title"],
            "department": r["department"],
            "match_score_pct": round(r["score"] * 100),
            "rationale": r["rationale"],
            "confirmed": bool(r["confirmed"]),
            "top_match": r["rank"] == 1,
        }
        for r in rows
    ]
    return {"mentee": {"id": mentee["id"], "name": mentee["name"]}, "matches": cards}


@app.get("/api/all-results")
def all_results(confirmed_only: bool = True):
    """
    Every match across everyone, sorted by score descending -- this is what
    powers the results table on the matching page: internal IDs, names, and
    emails for both parties, plus the match score and rationale.
    """
    conn = get_conn()
    where = "WHERE m.confirmed = 1" if confirmed_only else ""
    rows = conn.execute(
        f"""SELECT m.score, m.rationale, m.confirmed,
                   mentee.id AS mentee_id, mentee.name AS mentee_name, mentee.email AS mentee_email,
                   mentor.id AS mentor_id, mentor.name AS mentor_name, mentor.email AS mentor_email
            FROM matches m
            JOIN people mentee ON mentee.id = m.mentee_id
            JOIN people mentor ON mentor.id = m.mentor_id
            {where}
            ORDER BY m.score DESC"""
    ).fetchall()
    conn.close()

    return [
        {
            "mentee_id": r["mentee_id"],
            "mentee_name": r["mentee_name"],
            "mentee_email": r["mentee_email"] or "",
            "mentor_id": r["mentor_id"],
            "mentor_name": r["mentor_name"],
            "mentor_email": r["mentor_email"] or "",
            "match_score_pct": round(r["score"] * 100),
            "rationale": r["rationale"],
            "confirmed": bool(r["confirmed"]),
        }
        for r in rows
    ]


# Serve the frontend from the same app -- same origin as the API, so no CORS
# juggling. Mounted last so it doesn't shadow the /api/* routes above.
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
