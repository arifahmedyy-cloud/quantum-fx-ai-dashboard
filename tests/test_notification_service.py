"""Tests for src/services/notification_service.py.

Covers two confirmed audit bugs:
  1. notify_error() and notify_drawdown() existed on NotificationService but
     were never called anywhere in app.py's trading loop — a real exception
     or a daily-loss/consecutive-loss guard trip would never alert a
     configured Telegram/Discord channel. Fixed in app.py's trading_loop.
  2. Telegram's legacy Markdown parse_mode rejects any message with an
     unbalanced count of "_", "*", "[", "`" with HTTP 400 — and real Python
     exception messages very commonly contain an odd number of underscores
     (e.g. "name 'foo_bar' is not defined") purely by chance, silently
     dropping exactly the alerts that matter most. Fixed with a plain-text
     retry fallback in _send_telegram.
"""
import os
import sys
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.config import NotificationConfig
from src.services.notification_service import NotificationService
import src.services.notification_service as ns


class _FakeTelegramHandler(BaseHTTPRequestHandler):
    """Mimics real Telegram Bot API behavior: legacy Markdown mode returns
    HTTP 400 for unbalanced markdown entities, otherwise 200."""
    calls = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _FakeTelegramHandler.calls.append(body)
        text = body.get("text", "")
        if body.get("parse_mode") == "Markdown" and (
            text.count("_") % 2 != 0 or text.count("*") % 2 != 0
        ):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"ok": False, "description": "Bad Request: can't parse entities"}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_telegram_server(monkeypatch):
    _FakeTelegramHandler.calls = []
    server = HTTPServer(("127.0.0.1", 0), _FakeTelegramHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    real_post = ns.requests.post

    def fake_post(url, json=None, timeout=None):
        url = url.replace("https://api.telegram.org", f"http://127.0.0.1:{port}")
        return real_post(url, json=json, timeout=timeout)

    monkeypatch.setattr(ns.requests, "post", fake_post)
    yield _FakeTelegramHandler
    server.shutdown()


def _make_service():
    cfg = NotificationConfig(telegram_bot_token="FAKE:TOKEN", telegram_chat_id="123", enabled=True)
    return NotificationService(cfg)


class TestTelegramMarkdownFallback:
    def test_balanced_markdown_delivers_normally(self, fake_telegram_server):
        svc = _make_service()
        ok = svc._send_telegram("🚀 *Trade Executed*\nDirection: BUY")
        assert ok is True
        assert len(fake_telegram_server.calls) == 1  # no fallback needed

    def test_unbalanced_underscore_falls_back_to_plain_text(self, fake_telegram_server):
        """Regression test: a realistic Python exception message with an
        odd number of underscores must still be delivered."""
        svc = _make_service()
        message = "⚠️ *Bot Error*\nNameError: name 'foo_bar' is not defined"
        ok = svc._send_telegram(message)
        assert ok is True, "Alert was silently dropped instead of falling back to plain text"
        assert len(fake_telegram_server.calls) == 2
        assert fake_telegram_server.calls[0]["parse_mode"] == "Markdown"
        assert "parse_mode" not in fake_telegram_server.calls[1]

    def test_send_returns_true_end_to_end_via_send(self, fake_telegram_server):
        svc = _make_service()
        # send() should not raise even when enabled and the message has
        # unbalanced markdown.
        svc.send("Error: name 'x_y' is undefined")
        assert len(fake_telegram_server.calls) >= 1


class TestNotificationWiring:
    """Confirms app.py's trading loop actually calls notify_error /
    notify_drawdown — not just that the methods exist on the class."""

    def test_notify_error_and_notify_drawdown_referenced_in_trading_loop(self):
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path) as f:
            source = f.read()
        # Scope to trading_loop's body only.
        start = source.index("def trading_loop(")
        end = source.index("\ndef ", start + 1)
        loop_body = source[start:end]
        assert "notify_error(" in loop_body, (
            "notify_error() is not called anywhere in trading_loop — a real "
            "exception would never alert a configured Telegram/Discord channel."
        )
        assert "notify_drawdown(" in loop_body, (
            "notify_drawdown() is not called anywhere in trading_loop — a "
            "daily-loss/consecutive-loss guard trip would never alert."
        )
