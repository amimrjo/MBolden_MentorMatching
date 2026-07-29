"""
FastAPI backend for the mentor-matching prototype.

Endpoints:
  POST /api/rebuild-matches        -- recompute shortlists for every mentee (retrieval step)
  POST /api/confirm-assignments    -- run capacity-aware assignment over current shortlists
  GET  /api/mentees                -- list everyone who can request a match
  GET  /api/matches/{mentee_id}    -- shortlist for one mentee, shaped for the UI cards
"""
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .database import get_conn, init_db
from .embeddings import embed_text
from .matching import confirm_assignments, generate_shortlists
from .resume_parser import UnsupportedFileType, extract_text, guess_name_and_title

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


@app.post("/api/bulk-upload-resumes")
async def bulk_upload_resumes(
    pool: str = Form(...),  # 'mentor' | 'mentee' | 'both'
    default_mentor_capacity: int = Form(5),
    files: list[UploadFile] = File(...),
):
    """
    Bulk ingest for the three-bucket upload flow: one call per bucket
    (mentee resumes, mentor resumes, open-to-both resumes). Which bucket a
    file lands in IS its role_pool -- no per-file role classification needed,
    which sidesteps the "is this person a mentor or mentee" guessing problem
    entirely. Name/title are best-effort guessed from the resume text
    (see resume_parser.guess_name_and_title); department/seniority aren't
    guessed and stay blank unless a bulk-edit endpoint fills them in later.
    """
    if pool not in VALID_ROLE_POOLS:
        raise HTTPException(400, f"pool must be one of {VALID_ROLE_POOLS}")

    conn = get_conn()
    results = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            tmp_path = Path(tmp.name)

        try:
            text = extract_text(tmp_path)
        except UnsupportedFileType as e:
            results.append({"filename": upload.filename, "status": "skipped", "reason": str(e)})
            continue
        finally:
            tmp_path.unlink(missing_ok=True)

        if not text.strip():
            results.append({"filename": upload.filename, "status": "skipped", "reason": "no extractable text"})
            continue

        name, title = guess_name_and_title(text)
        capacity = default_mentor_capacity if pool in ("mentor", "both") else 0

        vector = embed_text(text)
        cur = conn.execute(
            """INSERT INTO people (name, title, department, seniority, role_pool, resume_text, mentor_capacity)
               VALUES (?, ?, '', '', ?, ?, ?)""",
            (name or upload.filename, title, pool, text, capacity),
        )
        conn.execute(
            "UPDATE people SET embedding = ? WHERE id = ?",
            (json.dumps(vector), cur.lastrowid),
        )
        results.append({
            "filename": upload.filename, "status": "ingested", "id": cur.lastrowid,
            "guessed_name": name, "guessed_title": title,
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
def confirm():
    confirmed, total = confirm_assignments()
    return {"confirmed": confirmed, "total_candidate_pairs": total}


@app.get("/api/mentees")
def list_mentees():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, title, department FROM people WHERE role_pool IN ('mentee','both') ORDER BY name"
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
