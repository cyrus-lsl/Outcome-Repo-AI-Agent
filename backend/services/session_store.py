"""In-memory session store for multi-turn agent context."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}

    def get_context(self, session_id: str) -> dict[str, Any]:
        return deepcopy(self._sessions.get(session_id, {}))

    def update(
        self,
        session_id: str,
        intent: str,
        user_query: str,
        response: dict[str, Any],
    ) -> None:
        session = self._sessions.setdefault(session_id, {})
        session['last_intent'] = intent
        session['last_query'] = user_query
        session['last_matched'] = response.get('matched', [])
        session['last_response_text'] = response.get('text', '')
        session['last_entities'] = response.get('entities', {})
        history = session.setdefault('history', [])
        history.append({'role': 'user', 'content': user_query})
        history.append({'role': 'assistant', 'content': response.get('text', '')})
        if len(history) > 20:
            session['history'] = history[-20:]

    def ensure_session(self, session_id: str) -> None:
        self._sessions.setdefault(session_id, {})


# Module-level singleton for Streamlit process
store = SessionStore()
