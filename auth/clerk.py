#!/usr/bin/env python3
"""Clerk session verification, without a network round-trip per request.

The dashboard signs users in with Clerk; the API trusts the resulting session
JWT. That trust is the whole security boundary, so it lives in one file with
one job: turn ``Authorization: Bearer <jwt>`` into a :class:`Principal`, or say
exactly why it will not.

Clerk signs with RS256 and publishes the public keys at
``<issuer>/.well-known/jwks.json``. We verify locally against that JWKS (cached
by PyJWT's ``PyJWKClient``), so an authenticated request costs one signature
check rather than a call to Clerk.

Configuration is entirely by environment, and unset configuration is a valid
state: :data:`Mode.OFF` is the default so a laptop and the test suite run
without credentials. A deployment opts in with::

    GATEKEEPER_AUTH_MODE=on            # "off" | "on"
    CLERK_ISSUER=https://your-app.clerk.accounts.dev
    CLERK_PUBLISHABLE_KEY=pk_test_...  # optional; the issuer can be derived
    CLERK_AUDIENCE=...                 # optional, only if tokens carry aud
    CLERK_AUTHORIZED_PARTIES=https://app.example.com  # optional azp allowlist

The issuer is derived from a publishable key when only that is set, because
Clerk's frontend API host is encoded in the key. ``CLERK_JWKS_JSON`` accepts a
literal JWKS document instead of a URL, which is what the tests use and what an
air-gapped deployment would mount.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jwt
from jwt import InvalidTokenError
from jwt.jwks_client import PyJWKClient

ENV_MODE = "GATEKEEPER_AUTH_MODE"
ENV_ISSUER = "CLERK_ISSUER"
ENV_PUBLISHABLE = "CLERK_PUBLISHABLE_KEY"
ENV_AUDIENCE = "CLERK_AUDIENCE"
ENV_PARTIES = "CLERK_AUTHORIZED_PARTIES"
ENV_JWKS_JSON = "CLERK_JWKS_JSON"

# Clock skew allowance, in seconds. Clerk issues short-lived tokens (60s by
# default) and clock drift between us and them is real; a token that expired
# ten seconds ago is not an attack worth logging out a user for.
LEEWAY_SECONDS = 10


class Mode(str, Enum):
    OFF = "off"
    ON = "on"


class AuthConfigError(RuntimeError):
    """Authentication was switched on but cannot be configured."""


class AuthError(Exception):
    """A credential was presented and rejected. ``status`` is the HTTP answer."""

    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, reduced to what the API actually uses."""

    user_id: str
    session_id: str = ""
    email: str = ""
    org_id: str = ""
    org_role: str = ""
    claims: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "email": self.email,
            "org_id": self.org_id,
            "org_role": self.org_role,
        }


def issuer_from_publishable_key(key: str) -> str:
    """Recover ``https://<frontend-api>$`` from a Clerk publishable key.

    The key is ``pk_test_``/``pk_live_`` followed by the base64 of the frontend
    API host plus a ``$`` sentinel. Deriving it here means one env var fewer and
    no chance of the issuer and the key disagreeing, which is otherwise a
    confusing 401 that looks like a bad token.
    """
    kind, environment, payload = (key.split("_", 2) + ["", ""])[:3]
    if kind not in {"pk", "sk"} or environment not in {"test", "live"} or not payload:
        raise AuthConfigError(
            f"not a Clerk key: expected pk_test_/pk_live_<base64 host>, got {key[:11]!r}..."
        )
    padded = payload + "=" * (-len(payload) % 4)
    try:
        host = base64.b64decode(padded).decode("utf-8").rstrip("$")
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthConfigError(f"publishable key payload is not a base64 host: {exc}") from None
    if not host:
        raise AuthConfigError("publishable key decodes to an empty host")
    return f"https://{host}" if not host.startswith("http") else host


@dataclass(frozen=True)
class Settings:
    """Everything verification needs, resolved once at startup."""

    mode: Mode = Mode.OFF
    issuer: str = ""
    audience: str = ""
    authorized_parties: tuple[str, ...] = ()
    jwks_url: str = ""
    jwks_json: str = ""
    publishable_key: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode is Mode.ON

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        raw_mode = (env.get(ENV_MODE) or "").strip().lower()
        publisher = (env.get(ENV_PUBLISHABLE) or "").strip()
        issuer = (env.get(ENV_ISSUER) or "").strip().rstrip("/")

        # Switching auth on by supplying only a key is the least surprising
        # behaviour: a deployment that sets CLERK_* clearly wants it enforced.
        if not raw_mode:
            raw_mode = Mode.ON.value if (issuer or publisher) else Mode.OFF.value
        try:
            mode = Mode(raw_mode)
        except ValueError:
            raise AuthConfigError(
                f"{ENV_MODE} must be one of {', '.join(m.value for m in Mode)}, got {raw_mode!r}"
            ) from None

        if mode is Mode.OFF:
            return cls(mode=mode, issuer=issuer, publishable_key=publisher)

        if not issuer and publisher:
            issuer = issuer_from_publishable_key(publisher)
        if not issuer:
            raise AuthConfigError(
                f"auth is on but neither {ENV_ISSUER} nor {ENV_PUBLISHABLE} is set; "
                "set one, or set "
                f"{ENV_MODE}=off"
            )
        if not issuer.startswith("https://"):
            raise AuthConfigError(f"{ENV_ISSUER} must be an https:// URL, got {issuer!r}")

        parties = tuple(
            p.strip().rstrip("/")
            for p in (env.get(ENV_PARTIES) or "").split(",")
            if p.strip()
        )
        return cls(
            mode=mode,
            issuer=issuer,
            audience=(env.get(ENV_AUDIENCE) or "").strip(),
            authorized_parties=parties,
            jwks_url=f"{issuer}/.well-known/jwks.json",
            jwks_json=(env.get(ENV_JWKS_JSON) or "").strip(),
            publishable_key=publisher,
        )


class _Jwks:
    """One cached JWKS reader, whichever way it was configured.

    ``PyJWKClient`` does its own caching and refresh-on-unknown-kid, which is
    exactly what key rotation requires. Constructing it lazily means importing
    this module never performs I/O.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: PyJWKClient | None = None
        self._static: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def key_for(self, token: str) -> Any:
        if self._settings.jwks_json:
            with self._lock:
                if self._static is None:
                    self._static = json.loads(self._settings.jwks_json)
            source = _StaticJwks(self._static)
            return source.key_for(token)
        with self._lock:
            if self._client is None:
                self._client = PyJWKClient(self._settings.jwks_url)
        try:
            return self._client.get_signing_key_from_jwt(token)
        except InvalidTokenError:
            raise
        except Exception as exc:  # network/parse failure, not a bad credential
            raise AuthError(
                f"identity provider unreachable while verifying the session: {type(exc).__name__}",
                status=503,
            ) from exc


class _StaticJwks:
    """A JWKS document held in memory: ``kid`` lookup without HTTP."""

    def __init__(self, document: dict[str, Any]) -> None:
        if not isinstance(document, dict) or not isinstance(document.get("keys"), list):
            raise AuthConfigError(f"{ENV_JWKS_JSON} must be a JWKS document with a 'keys' list")
        self._keys = {key.get("kid"): key for key in document["keys"] if isinstance(key, dict)}

    def key_for(self, token: str) -> Any:
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except InvalidTokenError as exc:
            raise AuthError(f"malformed token header: {exc}") from None
        if not kid or kid not in self._keys:
            raise AuthError(f"token signed with unknown key id {kid!r}")
        try:
            return jwt.PyJWK(self._keys[kid])
        except InvalidTokenError as exc:
            raise AuthError(f"JWKS entry {kid!r} is unusable: {exc}") from None


_jwks_cache: dict[str, _Jwks] = {}
_jwks_lock = threading.Lock()


def _jwks_for(settings: Settings) -> _Jwks:
    key = settings.jwks_json or settings.jwks_url
    with _jwks_lock:
        if key not in _jwks_cache:
            _jwks_cache[key] = _Jwks(settings)
        return _jwks_cache[key]


def clear_jwks_cache() -> None:
    """Drop cached key material (tests switch issuers between cases)."""
    with _jwks_lock:
        _jwks_cache.clear()


def verify(token: str, settings: Settings) -> Principal:
    """Verify a Clerk session JWT and return its principal.

    Rejections raise :class:`AuthError`. A rejection that is our fault (bad
    issuer URL, provider down) is a 503 so it is not mistaken for a user who
    needs to sign in again.
    """
    if not settings.enabled:
        raise AuthError("authentication is not configured", status=503)
    if not token:
        raise AuthError("missing bearer token")

    try:
        header = jwt.get_unverified_header(token)
    except InvalidTokenError as exc:
        raise AuthError(f"malformed token: {exc}") from None

    # Pin the algorithm. Without this, a token could claim "HS256" and be
    # verified with the public key as an HMAC secret -- the classic JWT
    # algorithm-confusion attack.
    if header.get("alg") != "RS256":
        raise AuthError(f"unsupported token algorithm {header.get('alg')!r}")

    key = _jwks_for(settings).key_for(token)

    options = {"require": ["exp", "iat", "iss", "sub"]}
    kwargs: dict[str, Any] = {
        "key": key.key,
        "algorithms": ["RS256"],
        "issuer": settings.issuer,
        "leeway": LEEWAY_SECONDS,
        "options": options,
    }
    if settings.audience:
        kwargs["audience"] = settings.audience

    try:
        claims = jwt.decode(token, **kwargs)
    except jwt.ExpiredSignatureError:
        raise AuthError("session expired; sign in again") from None
    except jwt.InvalidIssuerError:
        raise AuthError(f"token was not issued by {settings.issuer}") from None
    except jwt.InvalidAudienceError:
        raise AuthError("token is not addressed to this API") from None
    except InvalidTokenError as exc:
        raise AuthError(f"invalid token: {exc}") from None

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AuthError("token has no subject")

    if settings.authorized_parties:
        azp = claims.get("azp")
        if azp not in settings.authorized_parties:
            raise AuthError(f"token was issued to {azp!r}, which is not an authorized party")

    return Principal(
        user_id=subject,
        session_id=str(claims.get("sid") or ""),
        email=str(claims.get("email") or ""),
        org_id=str(claims.get("org_id") or ""),
        org_role=str(claims.get("org_role") or ""),
        claims=claims,
    )


__all__ = [
    "AuthConfigError",
    "AuthError",
    "Mode",
    "Principal",
    "Settings",
    "clear_jwks_cache",
    "issuer_from_publishable_key",
    "verify",
]
