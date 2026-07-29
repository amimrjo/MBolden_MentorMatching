"""
Bulk resume ingest. In production this is where parsed CVs/bios/org-chart
data would land (e.g. from an upload endpoint or HR system export). For the
prototype we load a sample JSON file standing in for "bulk resumes".
"""
import json
from pathlib import Path

from app.database import get_conn, init_db, save_embedding
from app.embeddings import embed_batch

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "resumes.json"


def seed():
    init_db()
    conn = get_conn()
    conn.execute("DELETE FROM matches")
    conn.execute("DELETE FROM people")

    people = json.loads(SAMPLE_PATH.read_text())
    texts = [p["resume_text"] for p in people]

    print(f"Embedding {len(texts)} resumes with all-MiniLM-L6-v2...")
    vectors = embed_batch(texts)

    for person, vector in zip(people, vectors):
        cur = conn.execute(
            """INSERT INTO people (name, title, department, seniority, role_pool, resume_text, mentor_capacity)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                person["name"], person["title"], person["department"], person["seniority"],
                person["role_pool"], person["resume_text"], person["mentor_capacity"],
            ),
        )
        save_embedding(conn, cur.lastrowid, vector)

    conn.commit()
    conn.close()
    print(f"Seeded {len(people)} people into {Path('mentor_matching.db').resolve()}")


if __name__ == "__main__":
    seed()
