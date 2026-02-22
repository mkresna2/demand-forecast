"""
Per-user session persistence using SQLite.
Stores raw data, trained models, and metrics per username.
"""

import base64
import sqlite3
import pickle
import json
import os
from pathlib import Path
from typing import Optional, Any

# Database path: same directory as this module
DB_PATH = Path(__file__).resolve().parent / "users.db"


def _json_default(obj: Any) -> Any:
    """Convert numpy/types to JSON-serializable Python types."""
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

SESSION_KEYS = (
    "ts_models", "ts_metrics", "ts_feat_cols", "ts_df", "daily",
    "m_cancel", "cancel_metrics", "m_los", "los_metrics", "raw", "expanded",
)


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
    """Create user_sessions table if it does not exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                username TEXT PRIMARY KEY,
                raw_blob BLOB,
                expanded_blob BLOB,
                daily_blob BLOB,
                ts_df_blob BLOB,
                ts_models_blob BLOB,
                ts_metrics_text TEXT,
                ts_feat_cols_text TEXT,
                m_cancel_blob BLOB,
                cancel_metrics_text TEXT,
                m_los_blob BLOB,
                los_metrics_text TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()


def _serialize_df(df) -> Optional[bytes]:
    if df is None:
        return None
    return pickle.dumps(df)


def _deserialize_df(blob: Optional[bytes]):
    if blob is None:
        return None
    return pickle.loads(blob)


def save_session(username: str, state: dict) -> None:
    """
    Persist session state for the given username.
    state: dict with keys matching SESSION_KEYS (e.g. from st.session_state).
    """
    import datetime
    init_db()
    now = datetime.datetime.utcnow().isoformat() + "Z"

    raw_blob = _serialize_df(state.get("raw"))
    expanded_blob = _serialize_df(state.get("expanded"))
    daily_blob = _serialize_df(state.get("daily"))
    ts_df_blob = _serialize_df(state.get("ts_df"))

    ts_models = state.get("ts_models")
    ts_models_blob = pickle.dumps(ts_models) if ts_models else None

    ts_metrics = state.get("ts_metrics")
    ts_metrics_text = json.dumps(ts_metrics, default=_json_default) if ts_metrics else None

    ts_feat_cols = state.get("ts_feat_cols")
    ts_feat_cols_text = json.dumps(ts_feat_cols) if ts_feat_cols else None

    m_cancel = state.get("m_cancel")
    m_cancel_blob = pickle.dumps(m_cancel) if m_cancel else None

    cancel_metrics = state.get("cancel_metrics")
    cancel_metrics_text = (
        base64.b64encode(pickle.dumps(cancel_metrics)).decode("ascii")
        if cancel_metrics else None
    )

    m_los = state.get("m_los")
    m_los_blob = pickle.dumps(m_los) if m_los else None

    los_metrics = state.get("los_metrics")
    los_metrics_text = (
        base64.b64encode(pickle.dumps(los_metrics)).decode("ascii")
        if los_metrics else None
    )

    with _get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO user_sessions (
                username, raw_blob, expanded_blob, daily_blob, ts_df_blob,
                ts_models_blob, ts_metrics_text, ts_feat_cols_text,
                m_cancel_blob, cancel_metrics_text, m_los_blob, los_metrics_text,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username,
            raw_blob, expanded_blob, daily_blob, ts_df_blob,
            ts_models_blob, ts_metrics_text, ts_feat_cols_text,
            m_cancel_blob, cancel_metrics_text, m_los_blob, los_metrics_text,
            now,
        ))
        conn.commit()


def load_session(username: str) -> Optional[dict]:
    """
    Load persisted session state for the given username.
    Returns a dict with SESSION_KEYS, or None if no session exists.
    """
    init_db()
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM user_sessions WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        return None

    state = {}
    state["raw"] = _deserialize_df(row["raw_blob"])
    state["expanded"] = _deserialize_df(row["expanded_blob"])
    state["daily"] = _deserialize_df(row["daily_blob"])
    state["ts_df"] = _deserialize_df(row["ts_df_blob"])
    state["ts_models"] = pickle.loads(row["ts_models_blob"]) if row["ts_models_blob"] else None
    state["ts_metrics"] = json.loads(row["ts_metrics_text"]) if row["ts_metrics_text"] else None
    state["ts_feat_cols"] = json.loads(row["ts_feat_cols_text"]) if row["ts_feat_cols_text"] else None
    state["m_cancel"] = pickle.loads(row["m_cancel_blob"]) if row["m_cancel_blob"] else None
    state["cancel_metrics"] = _load_metrics_text(row["cancel_metrics_text"])
    state["m_los"] = pickle.loads(row["m_los_blob"]) if row["m_los_blob"] else None
    state["los_metrics"] = _load_metrics_text(row["los_metrics_text"])
    return state


def _load_metrics_text(text: Optional[str]):
    """Load metrics from TEXT column: base64 pickle or legacy JSON."""
    if not text:
        return None
    # New format: base64-encoded pickle (no leading '{')
    if not text.strip().startswith("{"):
        return pickle.loads(base64.b64decode(text))
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def list_users() -> list[str]:
    """Return list of usernames that have saved sessions."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT username FROM user_sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [r[0] for r in rows]
