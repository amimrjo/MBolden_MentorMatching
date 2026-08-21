"""
Bulk resume ingest from three folders -- one per pool. No manifest needed:
which folder a file is in IS its role_pool. Name/title are best-effort
guessed from the resume text (see app/resume_parser.guess_name_and_title).

Usage:
    python ingest_resumes.py \\
        --mentor-dir sample_data/resumes/mentors \\
        --mentee-dir sample_data/resumes/mentees \\
        --both-dir   sample_data/resumes/both \\
        --default-mentor-capacity 5

Any of the three dirs can be omitted if that bucket is empty.
"""
import argparse
from pathlib import Path

from app.database import get_conn, init_db, save_embedding
from app.embeddings import embed_batch
from app.resume_parser import UnsupportedFileType, extract_text, guess_email, guess_name_and_title

SUPPORTED = {".pdf", ".docx", ".txt"}


def collect_folder(folder: Path, pool: str, default_mentor_capacity: int):
    rows = []
    if not folder or not folder.exists():
        return rows

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED:
            continue
        try:
            text = extract_text(file_path)
        except UnsupportedFileType as e:
            print(f"  skipping {file_path.name}: {e}")
            continue
        if not text.strip():
            print(f"  skipping {file_path.name}: no extractable text")
            continue

        name, title = guess_name_and_title(text)
        email = guess_email(text)
        capacity = default_mentor_capacity if pool in ("mentor", "both") else 0
        rows.append({
            "name": name or file_path.stem,
            "email": email,
            "title": title,
            "department": "",
            "seniority": "",
            "role_pool": pool,
            "mentor_capacity": capacity,
            "resume_text": text,
        })
        print(f"  parsed {file_path.name} -> name guess: '{name}' | title guess: '{title}' | email: '{email}'")
    return rows


def ingest(mentor_dir, mentee_dir, both_dir, default_mentor_capacity=5, reset=True):
    init_db()
    conn = get_conn()
    if reset:
        conn.execute("DELETE FROM matches")
        conn.execute("DELETE FROM people")

    people = []
    for folder, pool in [(mentor_dir, "mentor"), (mentee_dir, "mentee"), (both_dir, "both")]:
        if folder:
            print(f"Reading {pool} bucket: {folder}")
            people += collect_folder(Path(folder), pool, default_mentor_capacity)

    if not people:
        print("No valid resumes found. Nothing written.")
        return

    print(f"\nExtracted text for {len(people)} resumes. Embedding...")
    vectors = embed_batch([p["resume_text"] for p in people])

    for person, vector in zip(people, vectors):
        cur = conn.execute(
            """INSERT INTO people (name, email, title, department, seniority, role_pool, resume_text, mentor_capacity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                person["name"], person["email"], person["title"], person["department"], person["seniority"],
                person["role_pool"], person["resume_text"], person["mentor_capacity"],
            ),
        )
        save_embedding(conn, cur.lastrowid, vector)

    conn.commit()
    conn.close()
    print(f"Ingested {len(people)} people across mentor/mentee/both buckets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mentor-dir", default=None)
    parser.add_argument("--mentee-dir", default=None)
    parser.add_argument("--both-dir", default=None)
    parser.add_argument("--default-mentor-capacity", type=int, default=5)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    ingest(
        args.mentor_dir, args.mentee_dir, args.both_dir,
        default_mentor_capacity=args.default_mentor_capacity,
        reset=not args.no_reset,
    )
