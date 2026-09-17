#!/usr/bin/env python3
"""
W5.6: Outbound webhook module for Gatekeeper gate events.

Fires webhooks on:
  - commit_scored: every commit scored
  - rule_blocked: a block-severity rule fired
  - budget_breached: repo's rolling risk budget exceeded
  - drift_detected: model score distribution drift detected
  - model_promoted: a new model version was promoted

Features:
  - HMAC-SHA256 signature in X-Gatekeeper-Signature header
  - Retry with exponential backoff, max 3 attempts
  - Failures logged, never fatal — a webhook outage must not fail the gate
  - Versioned payload schema (docs/webhook.md)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WEBHOOK_VERSION = "1.0.0"
MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 9]  # seconds, exponential backoff


@dataclass
class WebhookPayload:
    """Versioned webhook payload."""
    version: str = WEBHOOK_VERSION
    event: str = ""  # commit_scored, rule_blocked, etc.
    timestamp: str = ""
    repo: str = ""
    commit_sha: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "event": self.event,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
            "repo": self.repo,
            "commit_sha": self.commit_sha,
            "data": self.data,
        }


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Sign payload with HMAC-SHA256. Returns hex digest."""
    return hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()


def fire_webhook(
    url: str,
    payload: WebhookPayload,
    secret: str = "",
    timeout: int = 10,
) -> bool:
    """Fire a webhook with HMAC signing and retry.

    Args:
        url: webhook endpoint URL
        payload: WebhookPayload to send
        secret: HMAC signing secret (from env WEBHOOK_SECRET)
        timeout: request timeout in seconds

    Returns:
        True if delivered, False if all retries failed
    """
    if not url:
        return False

    secret = secret or os.environ.get("WEBHOOK_SECRET", "")
    payload_dict = payload.to_dict()
    payload_bytes = json.dumps(payload_dict, default=str).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"Gatekeeper-Webhook/{WEBHOOK_VERSION}",
        "X-Gatekeeper-Event": payload.event,
        "X-Gatekeeper-Version": WEBHOOK_VERSION,
    }

    if secret:
        sig = sign_payload(payload_bytes, secret)
        headers["X-Gatekeeper-Signature"] = f"sha256={sig}"

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                data=payload_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if 200 <= status < 300:
                    return True
                last_error = f"HTTP {status}"
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:
            last_error = str(e)

        # Wait before retry (exponential backoff)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])

    # All retries failed — log but do not raise
    print(f"WEBHOOK FAILED after {MAX_RETRIES} attempts to {url}: {last_error}",
          file=sys.stderr)
    return False


def emit_commit_scored(
    url: str,
    repo: str,
    commit_sha: str,
    band: str,
    risk_score: float,
    rule_results: list[dict] = None,
    shap_top3: list[dict] = None,
    secret: str = "",
) -> bool:
    """Emit a commit_scored webhook event."""
    data = {
        "band": band,
        "risk_score": risk_score,
        "blocked": any(
            not r.get("passed", True) and r.get("severity") == "block"
            for r in (rule_results or [])
        ),
        "rules_fired": [
            {"rule": r["rule"], "severity": r["severity"], "message": r.get("message", "")}
            for r in (rule_results or []) if not r.get("passed", True)
        ],
        "shap_top3": shap_top3 or [],
    }

    event = "rule_blocked" if data["blocked"] else "commit_scored"
    payload = WebhookPayload(
        event=event,
        repo=repo,
        commit_sha=commit_sha,
        data=data,
    )
    return fire_webhook(url, payload, secret=secret)


def emit_budget_breached(
    url: str,
    repo: str,
    budget_pct: float,
    budget_cap: float,
    secret: str = "",
) -> bool:
    """Emit a budget_breached webhook event."""
    payload = WebhookPayload(
        event="budget_breached",
        repo=repo,
        data={
            "budget_pct": budget_pct,
            "budget_cap": budget_cap,
            "message": f"Repo {repo} at {budget_pct:.1f}% of {budget_cap:.0f}% budget",
        },
    )
    return fire_webhook(url, payload, secret=secret)


def emit_drift_detected(
    url: str,
    repo: str,
    metric: str,
    value: float,
    threshold: float,
    secret: str = "",
) -> bool:
    """Emit a drift_detected webhook event."""
    payload = WebhookPayload(
        event="drift_detected",
        repo=repo,
        data={
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "message": f"Drift detected on {metric}: {value:.4f} (threshold: {threshold:.4f})",
        },
    )
    return fire_webhook(url, payload, secret=secret)


def emit_model_promoted(
    url: str,
    repo: str,
    model_version: str,
    roc_auc: float,
    secret: str = "",
) -> bool:
    """Emit a model_promoted webhook event."""
    payload = WebhookPayload(
        event="model_promoted",
        repo=repo,
        data={
            "model_version": model_version,
            "roc_auc": roc_auc,
            "message": f"Model {model_version} promoted (ROC-AUC: {roc_auc:.4f})",
        },
    )
    return fire_webhook(url, payload, secret=secret)


# ── Local test receiver ─────────────────────────────────────────────

def run_test_receiver(port: int = 9876):
    """Run a simple HTTP server that receives and prints webhooks."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            sig_header = self.headers.get("X-Gatekeeper-Signature", "")
            event = self.headers.get("X-Gatekeeper-Event", "unknown")

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {"raw": body.decode("utf-8", errors="replace")}

            received.append({
                "event": event,
                "signature": sig_header,
                "data": data,
            })

            print(f"\n--- Webhook #{len(received)} ---")
            print(f"Event: {event}")
            print(f"Signature: {sig_header}")
            print(f"Payload: {json.dumps(data, indent=2, default=str)}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "received"}')

        def log_message(self, format, *args):
            pass  # Suppress default logging

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Webhook receiver listening on http://127.0.0.1:{port}")
    print("Waiting for webhooks...\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nReceived {len(received)} webhook(s) total.")
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test-receiver":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9876
        run_test_receiver(port)
    else:
        print("Usage:")
        print("  python -m ml.webhook test-receiver [port]  — run test receiver")
        print("  from ml.webhook import fire_webhook, emit_commit_scored, etc.")
