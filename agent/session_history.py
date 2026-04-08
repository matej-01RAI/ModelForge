"""Session history - persists conversation and build results across sessions."""

import os
import json
import time


SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(chat_history: list, plan: str = None, tokens: dict = None, workspace: str = None) -> str:
    """Save a session to disk. Returns the session file path."""
    _ensure_sessions_dir()

    session_id = f"session_{int(time.time())}"
    session_data = {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chat_history": chat_history,
        "plan": plan,
        "tokens": tokens,
        "workspace": workspace,
    }

    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2, default=str)

    return session_file


def list_sessions(limit: int = 10) -> list:
    """List recent sessions, newest first."""
    _ensure_sessions_dir()

    sessions = []
    for f in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if f.endswith(".json"):
            try:
                with open(os.path.join(SESSIONS_DIR, f), "r") as fh:
                    data = json.load(fh)
                    sessions.append({
                        "session_id": data.get("session_id", f),
                        "timestamp": data.get("timestamp", ""),
                        "workspace": data.get("workspace", ""),
                        "turns": len(data.get("chat_history", [])) // 2,
                        "file": f,
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        if len(sessions) >= limit:
            break

    return sessions


def load_session(session_id: str) -> dict:
    """Load a session by ID. Returns session data or empty dict."""
    _ensure_sessions_dir()

    # Try exact filename
    for filename in [f"{session_id}.json", session_id]:
        filepath = os.path.join(SESSIONS_DIR, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                return json.load(f)

    return {}


def format_session_list(sessions: list) -> str:
    """Format session list for display."""
    if not sessions:
        return "No saved sessions found."

    lines = ["**Saved Sessions:**\n"]
    for i, s in enumerate(sessions, 1):
        workspace = os.path.basename(s.get("workspace", "")) if s.get("workspace") else "—"
        lines.append(f"  {i}. `{s['session_id']}` — {s['timestamp']} — {s['turns']} turns — workspace: {workspace}")

    lines.append("\nUse `/load <session_id>` to restore a session.")
    return "\n".join(lines)
