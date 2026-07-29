"""
Matching engine.

Two separate concerns, kept apart on purpose:

1. shortlist generation  -- what a mentee SEES (top-k by embedding similarity,
   no capacity limit applied yet -- this is what renders in the UI cards).
2. capacity-aware assignment -- what actually gets CONFIRMED when a mentee
   requests a match, respecting each mentor's mentor_capacity (3-5 mentees).

"Open to both" people live in both pools simultaneously (per product decision):
they can appear as a mentor candidate for someone else's shortlist AND
receive their own shortlist of mentors, at the same time.
"""
import re
from collections import defaultdict

from .database import get_conn, load_embedding
from .embeddings import cosine_similarity

SHORTLIST_SIZE = 5
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "with", "for", "to",
    "is", "are", "was", "were", "at", "as", "by", "this", "that", "it",
    "years", "year", "experience", "working", "worked", "work",
}


def _mentor_pool(conn):
    return conn.execute(
        "SELECT * FROM people WHERE role_pool IN ('mentor', 'both')"
    ).fetchall()


def _mentee_pool(conn):
    return conn.execute(
        "SELECT * FROM people WHERE role_pool IN ('mentee', 'both')"
    ).fetchall()


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def generate_rationale(mentee_row, mentor_row) -> str:
    """
    Extractive rationale: surfaces the overlapping skills/keywords between
    the two resumes. Stands in for an LLM-generated ("RAG") explanation --
    swap this function's body for a call to a generation model later
    without touching any caller.
    """
    shared = _keywords(mentee_row["resume_text"]) & _keywords(mentor_row["resume_text"])
    # prefer longer, more specific terms over generic ones
    shared = sorted(shared, key=len, reverse=True)[:3]

    if mentee_row["department"] == mentor_row["department"]:
        dept_clause = f"same {mentee_row['department']} background"
    else:
        dept_clause = f"cross-department fit ({mentor_row['department']} -> {mentee_row['department']})"

    if shared:
        skills_clause = "shared focus on " + ", ".join(shared)
    else:
        skills_clause = "complementary experience profile"

    return f"{skills_clause}; {dept_clause}."


def generate_shortlists(top_k: int = SHORTLIST_SIZE):
    """
    For every mentee-seeking person, rank ALL mentor-eligible people by
    cosine similarity and keep the top_k. Writes results into `matches`
    with confirmed=0. This is the retrieval step of the RAG pipeline --
    generation (rationale) happens per-pair via generate_rationale().
    """
    conn = get_conn()
    conn.execute("DELETE FROM matches")

    mentors = _mentor_pool(conn)
    mentees = _mentee_pool(conn)

    for mentee in mentees:
        mentee_vec = load_embedding(mentee)
        scored = []
        for mentor in mentors:
            if mentor["id"] == mentee["id"]:
                continue  # a "both" person can't mentor themselves
            score = cosine_similarity(mentee_vec, load_embedding(mentor))
            scored.append((score, mentor))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_k]

        for rank, (score, mentor) in enumerate(top, start=1):
            rationale = generate_rationale(mentee, mentor)
            conn.execute(
                """INSERT INTO matches (mentee_id, mentor_id, score, rationale, rank)
                   VALUES (?, ?, ?, ?, ?)""",
                (mentee["id"], mentor["id"], score, rationale, rank),
            )

    conn.commit()
    conn.close()


def confirm_assignments():
    """
    Capacity-aware greedy assignment: walk all (mentee, mentor) shortlist
    pairs in descending score order, lock in a match if the mentee doesn't
    have one yet and the mentor is under mentor_capacity (3-5 mentees).
    This is what runs when matches move from "suggested" to "confirmed" --
    e.g. after a mentee clicks Request match, or on a scheduled batch run.
    """
    conn = get_conn()
    pairs = conn.execute(
        """SELECT m.*, p.mentor_capacity AS capacity
           FROM matches m JOIN people p ON p.id = m.mentor_id
           ORDER BY m.score DESC"""
    ).fetchall()

    mentor_load = defaultdict(int)
    mentee_assigned = set()
    confirmed_ids = []

    for pair in pairs:
        if pair["mentee_id"] in mentee_assigned:
            continue
        if mentor_load[pair["mentor_id"]] >= pair["capacity"]:
            continue
        mentor_load[pair["mentor_id"]] += 1
        mentee_assigned.add(pair["mentee_id"])
        confirmed_ids.append(pair["id"])

    if confirmed_ids:
        placeholders = ",".join("?" for _ in confirmed_ids)
        conn.execute(f"UPDATE matches SET confirmed = 1 WHERE id IN ({placeholders})", confirmed_ids)
    conn.commit()
    conn.close()
    return len(confirmed_ids), len(pairs)
