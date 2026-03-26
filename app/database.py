"""
=============================================================
app/database.py — SQLite Database Layer
=============================================================
Handles all persistence for the Medicine Adherence Tracker:
  - Users
  - Medicines
  - Adherence logs (time-series: user × medicine × timestamp × taken)
  - Time-series conversion for LSTM

Database is stored at: data/adherence.db
=============================================================
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ── Database path ─────────────────────────────────────────
DB_PATH = Path("data/adherence.db")


# ── Schema ────────────────────────────────────────────────

SCHEMA = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    UNIQUE,
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Medicines table (per user)
CREATE TABLE IF NOT EXISTS medicines (
    medicine_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    name          TEXT    NOT NULL,          -- e.g. "Metformin 500mg"
    type          TEXT    DEFAULT 'tablet',  -- tablet | strip | bottle | capsule
    dose_times    TEXT    DEFAULT '["08:00","20:00"]',  -- JSON list of scheduled times
    is_active     INTEGER DEFAULT 1,
    created_at    TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Adherence logs (core time-series table)
CREATE TABLE IF NOT EXISTS adherence_logs (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    medicine_id   INTEGER NOT NULL,
    medicine_name TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,          -- ISO format: 2024-01-15 08:05:32
    scheduled_for TEXT,                      -- ISO format of scheduled dose
    taken         INTEGER NOT NULL,          -- 1 = taken, 0 = missed
    confidence    REAL    DEFAULT 1.0,       -- detection confidence (0–1)
    notes         TEXT    DEFAULT '',
    created_at    TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (user_id)     REFERENCES users(user_id),
    FOREIGN KEY (medicine_id) REFERENCES medicines(medicine_id)
);

-- Detection events (raw YOLO detections)
CREATE TABLE IF NOT EXISTS detection_events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    timestamp     TEXT    NOT NULL,
    image_path    TEXT,
    detections    TEXT,                      -- JSON: list of {class, confidence, bbox}
    medicine_found INTEGER DEFAULT 0,
    created_at    TEXT    DEFAULT (datetime('now'))
);

-- Indexes for fast time-series queries
CREATE INDEX IF NOT EXISTS idx_logs_user    ON adherence_logs (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_med     ON adherence_logs (medicine_id, timestamp);
"""


# ── Database Connection ────────────────────────────────────

class Database:
    """
    SQLite database manager for the Medicine Adherence Tracker.
    
    Usage:
        db = Database()
        db.add_user("Alice", "alice@example.com")
        db.log_dose(user_id=1, medicine_id=1, taken=1)
        history = db.get_adherence_history(user_id=1, days=30)
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Get a database connection with row_factory for dict-like access."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        print(f"✅ Database initialized: {self.db_path}")

    # ═══════════════════════════════════════════════════════
    # USER MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def add_user(self, name: str, email: str = None) -> int:
        """Create a new user. Returns user_id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)",
                (name, email)
            )
            # Return existing if email already present
            if cur.rowcount == 0 and email:
                row = conn.execute("SELECT user_id FROM users WHERE email=?", (email,)).fetchone()
                return row["user_id"]
            return cur.lastrowid

    def get_user(self, user_id: int) -> Optional[Dict]:
        """Fetch user by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def list_users(self) -> List[Dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM users").fetchall()]

    # ═══════════════════════════════════════════════════════
    # MEDICINE MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def add_medicine(
        self,
        user_id:    int,
        name:       str,
        med_type:   str = "tablet",
        dose_times: List[str] = None,
    ) -> int:
        """
        Register a medicine for a user.
        
        Args:
            user_id:    User ID
            name:       Medicine name (e.g. "Paracetamol 500mg")
            med_type:   tablet | strip | bottle | capsule
            dose_times: List of HH:MM strings, e.g. ["08:00", "20:00"]
        
        Returns:
            medicine_id
        """
        times_json = json.dumps(dose_times or ["08:00"])
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO medicines (user_id, name, type, dose_times) VALUES (?,?,?,?)",
                (user_id, name, med_type, times_json)
            )
            return cur.lastrowid

    def get_medicines(self, user_id: int) -> List[Dict]:
        """Get all active medicines for a user."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM medicines WHERE user_id=? AND is_active=1",
                (user_id,)
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["dose_times"] = json.loads(d["dose_times"])
                result.append(d)
            return result

    # ═══════════════════════════════════════════════════════
    # ADHERENCE LOGGING (CORE)
    # ═══════════════════════════════════════════════════════

    def log_dose(
        self,
        user_id:      int,
        medicine_id:  int,
        taken:        int,                    # 1 = taken, 0 = missed
        medicine_name: str = "Medicine",
        confidence:   float = 1.0,
        notes:        str = "",
        timestamp:    str = None,             # Override timestamp (for testing)
    ) -> int:
        """
        Log a dose event (taken or missed).
        
        This is the core time-series data entry point.
        
        Returns:
            log_id
        """
        ts = timestamp or datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO adherence_logs
                   (user_id, medicine_id, medicine_name, timestamp, taken, confidence, notes)
                   VALUES (?,?,?,?,?,?,?)""",
                (user_id, medicine_id, medicine_name, ts, int(taken), confidence, notes)
            )
            return cur.lastrowid

    def log_detection_event(
        self,
        user_id:        int,
        detections:     List[Dict],
        medicine_found: bool,
        image_path:     str = None,
    ) -> int:
        """Log raw detection event (separate from dose log)."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO detection_events
                   (user_id, timestamp, image_path, detections, medicine_found)
                   VALUES (?,?,?,?,?)""",
                (
                    user_id,
                    datetime.now().isoformat(sep=" ", timespec="seconds"),
                    image_path,
                    json.dumps(detections),
                    int(medicine_found),
                )
            )
            return cur.lastrowid

    # ═══════════════════════════════════════════════════════
    # FETCH & QUERY
    # ═══════════════════════════════════════════════════════

    def get_adherence_history(
        self,
        user_id:     int,
        medicine_id: int = None,
        days:        int = 30,
    ) -> pd.DataFrame:
        """
        Fetch adherence history as a pandas DataFrame.
        
        Returns columns:
            log_id, user_id, medicine_id, medicine_name,
            timestamp (datetime), taken (int), confidence, notes
        """
        since = (datetime.now() - timedelta(days=days)).isoformat(sep=" ")
        query = """
            SELECT * FROM adherence_logs
            WHERE user_id=? AND timestamp >= ?
        """
        params = [user_id, since]
        
        if medicine_id:
            query += " AND medicine_id=?"
            params.append(medicine_id)
        
        query += " ORDER BY timestamp ASC"
        
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df

    def get_today_summary(self, user_id: int) -> Dict:
        """
        Get today's adherence summary for a user.
        
        Returns:
            {
                total_scheduled: int,
                total_taken: int,
                total_missed: int,
                adherence_pct: float,
                medicines: List[Dict]
            }
        """
        today = datetime.now().date().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT medicine_name, taken, COUNT(*) as cnt
                   FROM adherence_logs
                   WHERE user_id=? AND date(timestamp)=?
                   GROUP BY medicine_name, taken""",
                (user_id, today)
            ).fetchall()
        
        taken = sum(r["cnt"] for r in rows if r["taken"] == 1)
        missed = sum(r["cnt"] for r in rows if r["taken"] == 0)
        total = taken + missed
        
        return {
            "date":             today,
            "total_scheduled":  total,
            "total_taken":      taken,
            "total_missed":     missed,
            "adherence_pct":    round((taken / total * 100) if total > 0 else 0, 1),
        }

    # ═══════════════════════════════════════════════════════
    # TIME-SERIES CONVERSION (FOR LSTM)
    # ═══════════════════════════════════════════════════════

    def get_daily_sequence(
        self,
        user_id:     int,
        medicine_id: int,
        days:        int = 90,
    ) -> np.ndarray:
        """
        Convert adherence logs to a daily binary time-series for LSTM.
        
        For each day in the range:
            - 1 if at least one "taken" log exists
            - 0 if only "missed" logs exist or no logs exist
        
        Returns:
            np.ndarray of shape (days,) with values in {0, 1}
        
        Example:
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, ...]
        """
        df = self.get_adherence_history(user_id, medicine_id, days)
        
        # Create a date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)
        date_range = pd.date_range(start_date, end_date, freq="D")
        
        if df.empty:
            # Return all zeros if no history
            return np.zeros(len(date_range), dtype=np.float32)
        
        # Aggregate to daily: 1 if taken any time that day
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date")["taken"].max().reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        
        # Reindex to full date range, fill missing days with 0
        daily = daily.set_index("date").reindex(date_range, fill_value=0)
        
        return daily["taken"].values.astype(np.float32)

    def get_all_sequences_for_training(
        self,
        sequence_length: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build training dataset (X, y) from ALL users' logs.
        
        Sliding window approach:
            - Window: sequence_length consecutive days
            - Label: next day (1=taken, 0=missed)
        
        Args:
            sequence_length: Number of past days to use as input
        
        Returns:
            X: np.ndarray of shape (N, sequence_length, 1)
            y: np.ndarray of shape (N,)
        """
        users = self.list_users()
        X_list, y_list = [], []
        
        for user in users:
            uid = user["user_id"]
            medicines = self.get_medicines(uid)
            
            for med in medicines:
                mid = med["medicine_id"]
                sequence = self.get_daily_sequence(uid, mid, days=180)
                
                # Sliding window
                for i in range(len(sequence) - sequence_length):
                    window = sequence[i : i + sequence_length]
                    label  = sequence[i + sequence_length]
                    X_list.append(window)
                    y_list.append(label)
        
        if not X_list:
            print("⚠️  No training data found. Using synthetic data for demo.")
            return self._generate_synthetic_sequences(sequence_length, n_samples=500)
        
        X = np.array(X_list, dtype=np.float32).reshape(-1, sequence_length, 1)
        y = np.array(y_list, dtype=np.float32)
        
        print(f"✅ Built training dataset: X={X.shape}, y={y.shape}")
        return X, y

    def _generate_synthetic_sequences(
        self,
        seq_len: int = 14,
        n_samples: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic synthetic adherence sequences for initial LSTM training.
        Models realistic patterns: weekday adherence, habit streaks, etc.
        
        NOT random — uses structured patterns for meaningful training.
        """
        np.random.seed(42)
        X_list, y_list = [], []
        
        for _ in range(n_samples):
            # Random "adherence rate" per person: 60–95%
            base_rate = np.random.uniform(0.60, 0.95)
            
            # Pattern: streaks (good weeks, bad weeks)
            seq = np.zeros(seq_len + 1)
            for i in range(seq_len + 1):
                # More likely to take if took yesterday (habit)
                if i > 0 and seq[i - 1] == 1:
                    prob = min(base_rate + 0.10, 0.98)
                else:
                    prob = max(base_rate - 0.15, 0.40)
                seq[i] = 1 if np.random.random() < prob else 0
            
            X_list.append(seq[:seq_len])
            y_list.append(seq[seq_len])
        
        X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, 1)
        y = np.array(y_list, dtype=np.float32)
        
        print(f"📊 Generated synthetic dataset: X={X.shape}, y={y.shape}")
        return X, y

    # ═══════════════════════════════════════════════════════
    # SEEDING (for demo / testing)
    # ═══════════════════════════════════════════════════════

    def seed_demo_data(self, user_id: int = 1):
        """Insert realistic past 60-day adherence data for demo/testing."""
        print("🌱 Seeding demo data...")
        
        # Add demo user
        uid = self.add_user("Demo User", "demo@example.com")
        mid = self.add_medicine(uid, "Metformin 500mg", "tablet", ["08:00", "20:00"])
        mid2 = self.add_medicine(uid, "Vitamin D3", "tablet", ["09:00"])
        
        # Seed 60 days of realistic adherence
        np.random.seed(0)
        base = datetime.now() - timedelta(days=60)
        
        for day in range(60):
            date = base + timedelta(days=day)
            
            # Simulate realistic patterns (weekdays better)
            weekday = date.weekday()
            base_prob = 0.90 if weekday < 5 else 0.75
            
            for med_id, med_name in [(mid, "Metformin 500mg"), (mid2, "Vitamin D3")]:
                for hour in [8, 20] if med_id == mid else [9]:
                    ts = date.replace(hour=hour, minute=np.random.randint(0, 30))
                    taken = 1 if np.random.random() < base_prob else 0
                    self.log_dose(uid, med_id, taken, med_name,
                                  timestamp=ts.isoformat(sep=" ", timespec="seconds"))
        
        print(f"✅ Seeded demo data for user_id={uid}")
        return uid


# ── Module test ───────────────────────────────────────────
if __name__ == "__main__":
    db = Database()
    uid = db.seed_demo_data()
    
    print("\n📋 Today's Summary:")
    summary = db.get_today_summary(uid)
    for k, v in summary.items():
        print(f"   {k}: {v}")
    
    print("\n📈 30-day history (last 5 rows):")
    history = db.get_adherence_history(uid, days=30)
    print(history.tail())
    
    print("\n🔢 Daily sequence (last 14 days):")
    meds = db.get_medicines(uid)
    seq = db.get_daily_sequence(uid, meds[0]["medicine_id"], days=14)
    print(f"   {seq}")
    
    print("\n📦 Training dataset:")
    X, y = db.get_all_sequences_for_training(sequence_length=14)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Positive rate: {y.mean():.1%}")
