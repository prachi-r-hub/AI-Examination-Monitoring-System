import os
import sqlite3
import time
import config

DB_PATH = os.path.join(config.BASE_DIR, "database", "incidents.db")

class IncidentDatabase:
    """
    Lightweight SQLite Database for storing, retrieving, and updating examination monitoring incidents.
    
    Fields:
    - incident_id (INTEGER PRIMARY KEY)
    - timestamp (REAL)
    - time_formatted (TEXT)
    - event_type (TEXT)
    - object_name (TEXT)
    - person_id (TEXT)
    - confidence (REAL)
    - duration (REAL)
    - risk_score (INTEGER)
    - risk_level (TEXT)
    - evidence_path (TEXT)
    - review_status (TEXT: 'Unreviewed', 'Reviewed', 'Dismissed')
    - notes (TEXT)
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes incidents table schema and indexes if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    timestamp REAL,
                    time_formatted TEXT,
                    event_type TEXT NOT NULL,
                    object_name TEXT,
                    person_id TEXT,
                    confidence REAL,
                    duration REAL,
                    risk_score INTEGER,
                    risk_level TEXT,
                    evidence_path TEXT,
                    review_status TEXT DEFAULT 'Unreviewed',
                    notes TEXT DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON incidents(review_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON incidents(timestamp)")
            conn.commit()

    def insert_incident(self, incident: dict) -> int:
        """
        Inserts a confirmed incident record into database.
        Prevents duplicate insertions for the same continuous event track.
        
        Returns:
            db_incident_id (int): Primary key ID of inserted row (or existing row if duplicate)
        """
        if not incident:
            return -1

        track_id = incident.get("track_id")
        event_type = incident.get("event_type", "Observable Event")
        obj_name = incident.get("object", incident.get("object_name", "Item"))
        pid = str(incident.get("person_id")) if incident.get("person_id") is not None else "N/A"
        t_stamp = float(incident.get("timestamp", time.time()))
        t_fmt = incident.get("time_formatted", time.strftime('%H:%M:%S', time.gmtime(t_stamp)))

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Deduplication Check 1: Check if this continuous track/event has already been inserted
            if track_id is not None:
                cursor.execute("SELECT incident_id FROM incidents WHERE track_id = ? AND event_type = ?", (track_id, event_type))
                row = cursor.fetchone()
                if row:
                    return row["incident_id"]

            # Deduplication Check 2: Check window deduplication within 3 seconds
            cursor.execute(
                "SELECT incident_id FROM incidents WHERE event_type = ? AND object_name = ? AND person_id = ? AND ABS(timestamp - ?) < 3.0",
                (event_type, obj_name, pid, t_stamp)
            )
            row = cursor.fetchone()
            if row:
                return row["incident_id"]

            # Insert new record
            cursor.execute("""
                INSERT INTO incidents (
                    track_id, timestamp, time_formatted, event_type, object_name,
                    person_id, confidence, duration, risk_score, risk_level,
                    evidence_path, review_status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                track_id,
                t_stamp,
                t_fmt,
                event_type,
                obj_name,
                pid,
                float(incident.get("confidence", 0.0)),
                float(incident.get("duration", incident.get("duration_seconds", 0.0))),
                int(incident.get("risk_score", 0)),
                incident.get("risk_level", "LOW"),
                incident.get("evidence_frame", incident.get("evidence_path", "")),
                incident.get("review_status", "Unreviewed"),
                incident.get("notes", "")
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_incidents(self, review_status_filter: str = None) -> list:
        """Retrieves all incidents, optionally filtered by review_status ('Unreviewed', 'Reviewed', 'Dismissed')."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if review_status_filter:
                cursor.execute("SELECT * FROM incidents WHERE review_status = ? ORDER BY timestamp DESC", (review_status_filter,))
            else:
                cursor.execute("SELECT * FROM incidents ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_recent_incidents(self, limit: int = 10) -> list:
        """Retrieves the most recent N incidents ordered by timestamp DESC."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_incident_by_id(self, incident_id: int) -> dict:
        """Retrieves detailed incident record by incident_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_review_status(self, incident_id: int, new_status: str) -> bool:
        """Updates review status ('Unreviewed', 'Reviewed', 'Dismissed') for an incident."""
        valid_statuses = {"Unreviewed", "Reviewed", "Dismissed"}
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid review_status: '{new_status}'. Must be one of {valid_statuses}")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE incidents SET review_status = ? WHERE incident_id = ?", (new_status, incident_id))
            conn.commit()
            return cursor.rowcount > 0

    def add_notes(self, incident_id: int, notes: str) -> bool:
        """Updates or appends reviewer notes for an incident."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE incidents SET notes = ? WHERE incident_id = ?", (notes, incident_id))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self):
        """Clears all incidents from database (useful for resetting testing sessions)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM incidents")
            conn.commit()
