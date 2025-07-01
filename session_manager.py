#!/usr/bin/env python3
"""
Session management for conversation persistence and resumption.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config import HostConfig


class SessionManager:
    """Manages conversation sessions with persistence."""

    def __init__(
        self,
        sessions_dir: Optional[str] = None,
        config: Optional["HostConfig"] = None,
    ):
        if sessions_dir is None:
            from config import get_config_dir
            self.sessions_dir = get_config_dir() / "sessions"
        else:
            self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_id: Optional[str] = None
        self.current_messages: List[Dict[str, Any]] = []
        self.config = config

    def create_new_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        self.current_messages = []

        # Create session metadata
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": 0,
            "messages": [],
        }

        self._save_session(session_data)
        self._update_last_session(session_id)

        return session_id

    def continue_last_session(self) -> Optional[str]:
        """Continue the last session, return session ID if found."""
        last_session_id = self._get_last_session()
        if last_session_id:
            return self.resume_session(last_session_id)
        return None

    def resume_session(self, session_id: str) -> Optional[str]:
        """Resume a specific session by ID."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            self.current_session_id = session_id
            self.current_messages = session_data.get("messages", [])

            # Update last accessed
            session_data["last_updated"] = datetime.now().isoformat()
            self._save_session(session_data)
            self._update_last_session(session_id)

            return session_id

        except Exception as e:
            print(f"Error resuming session {session_id}: {e}")
            return None

    def add_message(self, message: Dict[str, Any]):
        """Add a message to the current session."""
        if not self.current_session_id:
            self.create_new_session()

        self.current_messages.append(message)
        self._save_current_session()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages from the current session."""
        return self.current_messages.copy()

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information about a session."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            # Get first few words of first user message for preview
            first_message = ""
            for msg in session_data.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    first_message = (
                        content[:50] + "..." if len(content) > 50 else content
                    )
                    break

            return {
                "session_id": session_id,
                "created_at": session_data.get("created_at"),
                "last_updated": session_data.get("last_updated"),
                "message_count": len(session_data.get("messages", [])),
                "first_message": first_message,
            }

        except Exception:
            return None

    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions with summary info."""
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            if session_file.name == "last_session.json":
                continue

            session_id = session_file.stem
            summary = self.get_session_summary(session_id)
            if summary:
                sessions.append(summary)

        # Sort by last updated, most recent first
        # Handle None values by ensuring we always have a sortable string
        sessions.sort(
            key=lambda x: (x.get("last_updated") or x.get("created_at") or "0"),
            reverse=True,
        )

        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if session_file.exists():
            try:
                session_file.unlink()

                # If this was the last session, clear it
                if self._get_last_session() == session_id:
                    self._clear_last_session()

                return True
            except Exception:
                return False

        return False

    def clear_all_sessions(self):
        """Clear all sessions."""
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                session_file.unlink()
            except Exception:
                pass

    def _save_current_session(self):
        """Save the current session to disk."""
        if not self.current_session_id:
            return

        session_data = {
            "session_id": self.current_session_id,
            "created_at": self._get_session_created_time(),
            "last_updated": datetime.now().isoformat(),
            "message_count": len(self.current_messages),
            "messages": self.current_messages,
        }

        self._save_session(session_data)

    def _save_session(self, session_data: Dict[str, Any]):
        """Save session data to file."""
        session_id = session_data["session_id"]
        session_file = self.sessions_dir / f"{session_id}.json"

        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session {session_id}: {e}")

    def _get_session_created_time(self) -> str:
        """Get the creation time of current session."""
        if not self.current_session_id:
            return datetime.now().isoformat()

        session_file = self.sessions_dir / f"{self.current_session_id}.json"
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                return session_data.get("created_at", datetime.now().isoformat())
            except Exception:
                pass

        return datetime.now().isoformat()

    def _update_last_session(self, session_id: str):
        """Update the last session pointer."""
        if self.config:
            # Use config-based storage for persistence
            self.config.set_last_session_id(session_id)
        else:
            # Fallback to file-based storage
            last_session_file = self.sessions_dir / "last_session.json"
            try:
                with open(last_session_file, "w", encoding="utf-8") as f:
                    json.dump({"last_session_id": session_id}, f)
            except Exception:
                pass

    def _get_last_session(self) -> Optional[str]:
        """Get the last session ID."""
        if self.config:
            # Use config-based storage
            return self.config.get_last_session_id()
        else:
            # Fallback to file-based storage
            last_session_file = self.sessions_dir / "last_session.json"
            if last_session_file.exists():
                try:
                    with open(last_session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("last_session_id")
                except Exception:
                    pass
            return None

    def _clear_last_session(self):
        """Clear the last session pointer."""
        if self.config:
            # Use config-based storage
            self.config.clear_last_session_id()
        else:
            # Fallback to file-based storage
            last_session_file = self.sessions_dir / "last_session.json"
            if last_session_file.exists():
                try:
                    last_session_file.unlink()
                except Exception:
                    pass


def main():
    """Test the session manager."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python session_manager.py <command>")
        print("Commands: create, list, continue, resume <id>, delete <id>, clear")
        return

    manager = SessionManager()
    command = sys.argv[1]

    if command == "create":
        session_id = manager.create_new_session()
        print(f"Created new session: {session_id}")

    elif command == "list":
        sessions = manager.list_sessions()
        if sessions:
            print("Recent sessions:")
            for i, session in enumerate(sessions, 1):
                created = session["created_at"][:19].replace("T", " ")
                print(
                    f"{i}. {session['session_id'][:8]}... ({session['message_count']} messages) - {created}"
                )
                if session["first_message"]:
                    print(f"   \"{session['first_message']}\"")
        else:
            print("No sessions found.")

    elif command == "continue":
        session_id = manager.continue_last_session()
        if session_id:
            print(f"Continuing session: {session_id}")
            print(f"Messages: {len(manager.get_messages())}")
        else:
            print("No previous session found.")

    elif command == "resume" and len(sys.argv) > 2:
        session_id = sys.argv[2]
        if manager.resume_session(session_id):
            print(f"Resumed session: {session_id}")
            print(f"Messages: {len(manager.get_messages())}")
        else:
            print(f"Session {session_id} not found.")

    elif command == "delete" and len(sys.argv) > 2:
        session_id = sys.argv[2]
        if manager.delete_session(session_id):
            print(f"Deleted session: {session_id}")
        else:
            print(f"Session {session_id} not found.")

    elif command == "clear":
        manager.clear_all_sessions()
        print("All sessions cleared.")

    else:
        print("Unknown command or missing arguments.")


if __name__ == "__main__":
    main()
