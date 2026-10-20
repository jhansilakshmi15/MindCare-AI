"""
MindCare AI - SQLite Database Interface
=========================================
Manages persistent storage for optional user accounts, screening records,
and anonymous session history using parameterized SQL queries.

Privacy by Design:
- Anonymized guest sessions by default via session UUID.
- Explicit user consent required before persisting records.
- Complete data removal functionality ('Delete History').
- Passwords stored securely via PBKDF2/SHA256 password hashing.
"""

import os
import sqlite3
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "app.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # User accounts (optional)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Screening records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screening_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
            age INTEGER,
            gender TEXT,
            family_history TEXT,
            work_interfere TEXT,
            benefits TEXT,
            care_options TEXT,
            anonymity TEXT,
            leave TEXT,
            stress_level INTEGER,
            sleep_quality TEXT,
            workload TEXT,
            prediction_label INTEGER,
            prediction_prob REAL,
            risk_tier TEXT,
            consent_saved INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

def create_user(username, password):
    """Registers a new user with secure password hash."""
    username = username.strip()
    if not username or not password:
        return None, "Username and password cannot be empty."
    if len(username) < 3:
        return None, "Username must be at least 3 characters."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."

    pwd_hash = generate_password_hash(password)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, pwd_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError:
        return None, "Username already exists. Please choose another."
    except Exception as e:
        return None, str(e)

def verify_user(username, password):
    """Verifies credentials and returns user dict if valid."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return dict(user)
    return None

def save_screening(record_dict):
    """
    Saves a screening record only if consent_saved is True.
    Uses parameterized query to prevent SQL injection.
    """
    if not record_dict.get("consent_saved", False):
        return None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO screening_records (
            user_id, session_id, age, gender, family_history,
            work_interfere, benefits, care_options, anonymity, leave,
            stress_level, sleep_quality, workload,
            prediction_label, prediction_prob, risk_tier, consent_saved
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record_dict.get("user_id"),
        record_dict.get("session_id"),
        record_dict.get("age"),
        record_dict.get("gender"),
        record_dict.get("family_history"),
        record_dict.get("work_interfere"),
        record_dict.get("benefits"),
        record_dict.get("care_options"),
        record_dict.get("anonymity"),
        record_dict.get("leave"),
        record_dict.get("stress_level"),
        record_dict.get("sleep_quality"),
        record_dict.get("workload"),
        record_dict.get("prediction_label"),
        record_dict.get("prediction_prob"),
        record_dict.get("risk_tier"),
        1 if record_dict.get("consent_saved") else 0
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_screenings_for_user_or_session(user_id=None, session_id=None):
    """Retrieves chronological screening records for a specific user or guest session."""
    conn = get_db_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT * FROM screening_records
            WHERE user_id = ?
            ORDER BY created_at ASC
        """, (user_id,))
    elif session_id:
        cursor.execute("""
            SELECT * FROM screening_records
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
    else:
        conn.close()
        return []

    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_user_screenings(user_id=None, session_id=None):
    """Deletes saved screening records for the requesting user or session."""
    conn = get_db_connection()
    cursor = conn.cursor()

    deleted_count = 0
    if user_id:
        cursor.execute("DELETE FROM screening_records WHERE user_id = ?", (user_id,))
        deleted_count = cursor.rowcount
    elif session_id:
        cursor.execute("DELETE FROM screening_records WHERE session_id = ?", (session_id,))
        deleted_count = cursor.rowcount

    conn.commit()
    conn.close()
    return deleted_count

def get_aggregate_admin_stats():
    """
    Computes aggregate metrics across local demo screenings without exposing
    individual identities or private responses.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total_count FROM screening_records")
    total_screenings = cursor.fetchone()["total_count"]

    cursor.execute("""
        SELECT risk_tier, COUNT(*) as count
        FROM screening_records
        GROUP BY risk_tier
    """)
    tier_distribution = {row["risk_tier"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT prediction_label, COUNT(*) as count
        FROM screening_records
        GROUP BY prediction_label
    """)
    label_distribution = {row["prediction_label"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("SELECT COUNT(*) as user_count FROM users")
    total_users = cursor.fetchone()["user_count"]

    conn.close()

    return {
        "total_screenings": total_screenings,
        "tier_distribution": tier_distribution,
        "label_distribution": label_distribution,
        "total_registered_users": total_users
    }

# Alias for route convenience
delete_screenings = delete_user_screenings
