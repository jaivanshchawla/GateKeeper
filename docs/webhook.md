# Gatekeeper Outbound Webhooks

Gatekeeper fires outbound webhooks on gate events. Configure webhook URLs in your `.gatekeeper.yml` or via the dashboard.

## Events

| Event | Description |
|-------|-------------|
| `commit_scored` | Every commit that is scored |
| `rule_blocked` | A block-severity rule fired |
| `budget_breached` | Repo's rolling risk budget exceeded |
| `drift_detected` | Model score distribution drift detected |
| `model_promoted` | A new model version was promoted |

## Payload Schema (v1.0.0)

```json
{
  "version": "1.0.0",
  "event": "commit_scored",
  "timestamp": "2026-09-17T12:00:00Z",
  "repo": "owner/repo",
  "commit_sha": "abc123def456",
  "data": { ... }
}
```

### `commit_scored` / `rule_blocked`

```json
{
  "version": "1.0.0",
  "event": "commit_scored",
  "timestamp": "2026-09-17T12:00:00Z",
  "repo": "django/django",
  "commit_sha": "a1b2c3d4e5f6",
  "data": {
    "band": "low|medium|high",
    "risk_score": 0.4523,
    "blocked": false,
    "rules_fired": [
      {
        "rule": "no_tests",
        "severity": "warn",
        "message": "Code changes without test changes"
      }
    ],
    "shap_top3": [
      {
        "feature": "days_since_last_change_max",
        "shap_value": 0.12,
        "human_readable": "avg days since touched files last changed: 42"
      }
    ]
  }
}
```

> Note: `rule_blocked` has the same payload as `commit_scored` but is emitted when `blocked: true`.

### `budget_breached`

```json
{
  "version": "1.0.0",
  "event": "budget_breached",
  "timestamp": "2026-09-17T12:00:00Z",
  "repo": "django/django",
  "commit_sha": "",
  "data": {
    "budget_pct": 27.5,
    "budget_cap": 25.0,
    "message": "Repo django/django at 27.5% of 25% budget"
  }
}
```

### `drift_detected`

```json
{
  "version": "1.0.0",
  "event": "drift_detected",
  "timestamp": "2026-09-17T12:00:00Z",
  "repo": "react/react",
  "commit_sha": "",
  "data": {
    "metric": "high_band_share",
    "value": 0.178,
    "threshold": 0.15,
    "message": "Drift detected on high_band_share: 0.1780 (threshold: 0.1500)"
  }
}
```

### `model_promoted`

```json
{
  "version": "1.0.0",
  "event": "model_promoted",
  "timestamp": "2026-09-17T12:00:00Z",
  "repo": "",
  "commit_sha": "",
  "data": {
    "model_version": "v9",
    "roc_auc": 0.7885,
    "message": "Model v9 promoted (ROC-AUC: 0.7885)"
  }
}
```

## Authentication

Every webhook request includes an HMAC-SHA256 signature header:

```
X-Gatekeeper-Signature: sha256=<hex-digest>
```

To verify:

```python
import hmac, hashlib

def verify_signature(payload_bytes: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

Set the `WEBHOOK_SECRET` environment variable on both the sender and receiver.

## Headers

| Header | Value |
|--------|-------|
| `Content-Type` | `application/json` |
| `User-Agent` | `Gatekeeper-Webhook/1.0.0` |
| `X-Gatekeeper-Event` | Event type (e.g. `commit_scored`) |
| `X-Gatekeeper-Version` | `1.0.0` |
| `X-Gatekeeper-Signature` | `sha256=<hmac-hex>` |

## Retry Behavior

- Max 3 attempts per webhook delivery
- Exponential backoff: 1s, 3s, 9s
- Failures are logged to stderr but never raise — a webhook outage must not fail the gate
- HTTP 2xx is considered success; anything else triggers a retry

## Configuration

```yaml
# .gatekeeper.yml
webhooks:
  enabled: true
  url: https://your-webhook-endpoint.com/gatekeeper
  secret: ${WEBHOOK_SECRET}  # or set via env var
  events:
    - commit_scored
    - rule_blocked
    - budget_breached
    - drift_detected
    - model_promoted
```
