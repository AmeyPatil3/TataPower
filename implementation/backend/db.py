"""
SQLite Database interface and schema management for Tata Power Load Analysis Platform.
Preserves canonical observations, EconML ATE/HTE causal effects, XAI attributions,
DTW similarity matches, documents, web context, and LLM explanations.
"""

import os
import sqlite3
import json
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "tatapower.db"
)

SCHEMA_SQL = """
-- 1. Canonical 15-Minute Observations
CREATE TABLE IF NOT EXISTS load_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT NOT NULL UNIQUE,     -- 'YYYY-MM-DD HH:MM:SS+05:30'
    date TEXT NOT NULL,                 -- 'YYYY-MM-DD'
    wdLoad REAL NOT NULL,               -- Megawatts (MW)
    temp REAL NOT NULL,                 -- Celsius
    rh REAL NOT NULL,                   -- Relative Humidity (%)
    hour INTEGER NOT NULL,              -- 0 - 23
    day_of_week INTEGER NOT NULL,       -- 0 (Monday) - 6 (Sunday)
    weekend INTEGER NOT NULL,           -- 0 or 1
    holiday INTEGER NOT NULL,           -- 0 or 1
    festival INTEGER NOT NULL,          -- 0 or 1
    office_hours INTEGER NOT NULL,      -- 0 or 1
    lag_load_15min REAL,                -- Previous 15-min load
    lag_load_prev_day REAL,             -- Load exactly 24 hours prior
    season TEXT NOT NULL,               -- 'Winter', 'Summer', 'Monsoon', 'Autumn'
    year INTEGER NOT NULL,              -- e.g. 2024
    temp_hte REAL,                      -- Tuple-level continuous temperature HTE from EconML CausalForestDML
    rh_hte REAL,                        -- Tuple-level continuous humidity HTE from EconML CausalForestDML
    weekend_hte REAL,                   -- Tuple-level weekend calendar HTE
    holiday_hte REAL,                   -- Tuple-level holiday calendar HTE
    festival_hte REAL,                  -- Tuple-level festival calendar HTE
    office_hte REAL                     -- Tuple-level office hours calendar HTE
);

CREATE INDEX IF NOT EXISTS idx_load_datetime ON load_data(datetime);
CREATE INDEX IF NOT EXISTS idx_load_date ON load_data(date);
CREATE INDEX IF NOT EXISTS idx_load_season ON load_data(season);
CREATE INDEX IF NOT EXISTS idx_load_calendar ON load_data(weekend, holiday, festival);
CREATE INDEX IF NOT EXISTS idx_load_hour ON load_data(hour);

-- 2. Causal Estimates: ATE and Diurnal Hourly HTE
CREATE TABLE IF NOT EXISTS causal_effects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    season TEXT NOT NULL,              -- 'Winter', 'Summer', 'Monsoon', 'Autumn'
    treatment TEXT NOT NULL,           -- 'temp', 'rh', 'festival', 'holiday', 'weekend', 'office_hours'
    effect_type TEXT NOT NULL,         -- 'ATE' (Overall Average) or 'HTE' (Hourly Heterogeneous)
    hour INTEGER,                      -- NULL for ATE; 0 - 23 for HTE
    effect_value REAL NOT NULL,        -- Estimated beta coefficient
    ci_lower REAL,                     -- 95% CI lower bound
    ci_upper REAL,                     -- 95% CI upper bound
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_causal_lookup ON causal_effects(season, treatment, effect_type, hour);

-- 3. XAI Attributions (SHAP and LIME)
CREATE TABLE IF NOT EXISTS xai_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    season TEXT NOT NULL,
    model_type TEXT NOT NULL,          -- 'xgboost'
    explanation_type TEXT NOT NULL,    -- 'SHAP_GLOBAL', 'SHAP_LOCAL', 'LIME'
    feature TEXT NOT NULL,             -- 'temp', 'rh', 'festival', etc.
    importance REAL,                   -- Mean absolute SHAP value
    observation_time TEXT,             -- Specific datetime for local explanations
    attribution_value REAL,            -- SHAP value or LIME weight
    actual_prediction REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_xai_lookup ON xai_results(season, explanation_type);

-- 4. DTW Similarity Matches Cache
CREATE TABLE IF NOT EXISTS dtw_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_date TEXT NOT NULL,          -- 'YYYY-MM-DD'
    historical_date TEXT NOT NULL,     -- 'YYYY-MM-DD'
    season TEXT NOT NULL,
    distance REAL NOT NULL,            -- Normalized DTW distance
    rank INTEGER NOT NULL,             -- 1 (closest), 2, 3...
    sequence_type TEXT NOT NULL,       -- 'univariate_load' or 'multivariate_weather'
    is_same_calendar_type INTEGER,     -- 1 if both share calendar profile
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(query_date, historical_date, sequence_type)
);

CREATE INDEX IF NOT EXISTS idx_dtw_query ON dtw_matches(query_date, rank);

-- 5. RAG Document Corpus
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source TEXT NOT NULL,              -- 'research_paper', 'synopsis', 'pbl_report'
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. External Web Search Evidence Cache
CREATE TABLE IF NOT EXISTS web_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    title TEXT,
    url TEXT,
    snippet TEXT,
    source_date TEXT,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Model Predictions & Residuals
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    datetime TEXT NOT NULL,
    actual_load REAL NOT NULL,
    predicted_load REAL NOT NULL,
    residual REAL NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. LLM Grounded Explanations Audit Log
CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    model_name TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """Context manager for SQLite connection with dictionary row access and WAL mode."""
    target_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conn = sqlite3.connect(target_path, timeout=15.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None):
    """Initializes the database schema and indexes."""
    with get_db_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def get_available_dates(season: Optional[str] = None, db_path: Optional[str] = None) -> List[str]:
    """Returns sorted unique dates available in the database."""
    with get_db_connection(db_path) as conn:
        if season:
            cursor = conn.execute(
                "SELECT DISTINCT date FROM load_data WHERE season = ? ORDER BY date",
                (season,)
            )
        else:
            cursor = conn.execute("SELECT DISTINCT date FROM load_data ORDER BY date")
        return [row["date"] for row in cursor.fetchall()]


def get_daily_records(date_str: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns all 15-minute observations for a given date sorted chronologically."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM load_data
            WHERE date = ?
            ORDER BY datetime ASC
            """,
            (date_str,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_causal_effects(season: str, effect_type: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves causal estimates (ATE or HTE) for a given season."""
    with get_db_connection(db_path) as conn:
        if effect_type:
            cursor = conn.execute(
                """
                SELECT * FROM causal_effects
                WHERE season = ? AND effect_type = ?
                ORDER BY treatment, hour
                """,
                (season, effect_type)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM causal_effects
                WHERE season = ?
                ORDER BY effect_type, treatment, hour
                """,
                (season,)
            )
        return [dict(row) for row in cursor.fetchall()]


def get_hourly_hte(season: str, treatment: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves the 24-hour diurnal HTE curve for a specific season and treatment."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT hour, effect_value, ci_lower, ci_upper
            FROM causal_effects
            WHERE season = ? AND treatment = ? AND effect_type = 'HTE'
            ORDER BY hour ASC
            """,
            (season, treatment)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_xai_results(season: str, explanation_type: str = "SHAP_GLOBAL", db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves XAI importance or attribution records."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM xai_results
            WHERE season = ? AND explanation_type = ?
            ORDER BY importance DESC
            """,
            (season, explanation_type)
        )
        return [dict(row) for row in cursor.fetchall()]


def insert_explanation(query: str, answer: str, evidence: Dict[str, Any], model_name: str = "gemma-4-e2b", db_path: Optional[str] = None) -> int:
    """Stores an LLM grounded explanation for auditability."""
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO explanations (query, answer, evidence_json, model_name)
            VALUES (?, ?, ?, ?)
            """,
            (query, answer, json.dumps(evidence), model_name)
        )
        return cursor.lastrowid


def get_observation_tuple(date_str: str, hour: Optional[int] = None, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves specific observation record with tuple-level HTEs."""
    with get_db_connection(db_path) as conn:
        if hour is not None:
            cursor = conn.execute(
                """
                SELECT * FROM load_data
                WHERE date = ? AND hour = ?
                ORDER BY datetime ASC
                LIMIT 1
                """,
                (date_str, hour)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM load_data
                WHERE date = ?
                ORDER BY wdLoad DESC
                LIMIT 1
                """,
                (date_str,)
            )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_day_hourly_hte(date_str: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Computes hourly average of row-level treatment effects for a specific day
    matching sociowinter_v2.ipynb Cell 13:
    hour_effect_day = day_df.groupby("hour")[["Temperature", "Relative Humidity", "Weekend", "Holiday", "Festival", "Office Hours"]].mean()
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT hour,
                   ROUND(AVG(temp_hte), 4) as temp_hte,
                   ROUND(AVG(rh_hte), 4) as rh_hte,
                   ROUND(AVG(weekend_hte), 4) as weekend_hte,
                   ROUND(AVG(holiday_hte), 4) as holiday_hte,
                   ROUND(AVG(festival_hte), 4) as festival_hte,
                   ROUND(AVG(office_hte), 4) as office_hte
            FROM load_data
            WHERE date = ?
            GROUP BY hour
            ORDER BY hour ASC
            """,
            (date_str,)
        )
        return [dict(row) for row in cursor.fetchall()]

