"""
Relational schema for the mentor-matching prototype.

people        one row per parsed resume, tagged with which pool(s) they belong to
matches       generated shortlist entries (mentee_id, mentor_id, score, rationale)
"""
import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "mentor_matching.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            title TEXT,
            department TEXT,
            seniority TEXT,                 -- junior / mid / senior / staff+
            role_pool TEXT NOT NULL,        -- 'mentor' | 'mentee' | 'both'
            resume_text TEXT NOT NULL,      -- CV / bio text used for embedding
            embedding BLOB,                 -- JSON-encoded float list
            mentor_capacity INTEGER DEFAULT 5   -- max mentees this person can take (if mentoring)
        );

        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mentee_id INTEGER NOT NULL REFERENCES people(id),
            mentor_id INTEGER NOT NULL REFERENCES people(id),
            score REAL NOT NULL,            -- cosine similarity, 0-1
            rationale TEXT NOT NULL,        -- short human-readable explanation
            rank INTEGER NOT NULL,          -- 1 = best match for this mentee
            confirmed INTEGER DEFAULT 0     -- 1 once mentor capacity assignment locks it in
        );
        """
    )
    conn.commit()
    conn.close()


def save_embedding(conn, person_id, vector):
    conn.execute(
        "UPDATE people SET embedding = ? WHERE id = ?",
        (json.dumps(vector), person_id),
    )


def load_embedding(row):
    return json.loads(row["embedding"])
