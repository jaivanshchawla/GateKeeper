#!/usr/bin/env python3
"""Clerk session verification, proven without a Clerk account.

Every case here is exercised against a locally generated RSA keypair published
as a JWKS document, so the test suite never needs network access, a Clerk
tenant, or a real user: the code path a deployed API takes -- parse the header,
look the key up by ``kid``, verify RS256, check ``iss``/``exp``/``sub`` -- is the
same one under test.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth import (
    AuthConfigError,
    AuthError,
    ClerkAuthMiddleware,
    Mode,
    Settings,
    clear_jwks_cache,
    is_public,
    issuer_from_publishable_key,
    verify,
)

ISSUER = "https://gatekeeper-test.clerk.accounts.dev"
KID = "ins_2abcDEF"


def _b64(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.fixture(scope="module")
def keypair() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks_json(keypair: rsa.RSAPrivateKey) -> str:
    numbers = keypair.public_key().public_numbers()
    return json.dumps(
        {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": KID,
                    "use": "sig",
                    "alg": "RS256",
                    "n": _b64(numbers.n),
                    "e": _b64(numbers.e),
                }
            ]
        }
    )


@pytest.fixture()
def settings(jwks_json: str) -> Settings:
    clear_jwks_cache()
    return Settings(mode=Mode.ON, issuer=ISSUER, jwks_json=jwks_json)


def make_token(
    keypair: rsa.RSAPrivateKey,
    *,
    issuer: str = ISSUER,
    subject: str = "user_2xYz",
    kid: str = KID,
    algorithm: str = "RS256",
    expires_in: int = 60,
    claims: dict[str, Any] | None = None,
    secret: Any = None,
) -> str:
    now = int(time.time())
    payload = {
        "sub": subject,
        "iss": issuer,
        "iat": now - 5,
        "nbf": now - 5,
        "exp": now + expires_in,
        "sid": "sess_123",
        "azp": "https://app.example.com",
    }
    payload.update(claims or {})
    return jwt.encode(
        payload,
        secret if secret is not None else keypair,
        algorithm=algorithm,
        headers={"kid": kid},
    )


# ── Settings ────────────────────────────────────────────────────────────────


def test_auth_is_off_when_nothing_is_configured():
    settings = Settings.from_env({})
    assert settings.mode is Mode.OFF
    assert settings.enabled is False


def test_a_clerk_key_alone_switches_auth_on_and_derives_the_issuer():
    host = "example-1234.clerk.accounts.dev"
    key = "pk_test_" + base64.b64encode(f"{host}$".encode()).decode().rstrip("=")
    settings = Settings.from_env({"CLERK_PUBLISHABLE_KEY": key})
    assert settings.enabled is True
    assert settings.issuer == f"https://{host}"
    assert settings.jwks_url == f"https://{host}/.well-known/jwks.json"


def test_issuer_and_key_are_reduced_to_one_issuer():
    assert issuer_from_publishable_key("pk_live_" + base64.b64encode(b"a.b.com$").decode()) == (
        "https://a.b.com"
    )
    with pytest.raises(AuthConfigError):
        issuer_from_publishable_key("not-a-clerk-key")
    with pytest.raises(AuthConfigError):
        issuer_from_publishable_key("pk_test_!!!!")


def test_auth_on_without_an_issuer_is_a_startup_error():
    with pytest.raises(AuthConfigError) as excinfo:
        Settings.from_env({"GATEKEEPER_AUTH_MODE": "on"})
    assert "CLERK_ISSUER" in str(excinfo.value)


def test_an_unparseable_mode_is_a_startup_error():
    with pytest.raises(AuthConfigError):
        Settings.from_env({"GATEKEEPER_AUTH_MODE": "sometimes"})


# ── Verification ────────────────────────────────────────────────────────────


def test_a_valid_token_becomes_a_principal(keypair, settings):
    principal = verify(make_token(keypair), settings)
    assert principal.user_id == "user_2xYz"
    assert principal.session_id == "sess_123"
    assert principal.to_dict()["user_id"] == "user_2xYz"


def test_optional_claims_are_carried(keypair, settings):
    token = make_token(
        keypair,
        claims={"email": "dev@example.com", "org_id": "org_1", "org_role": "admin"},
    )
    principal = verify(token, settings)
    assert (principal.email, principal.org_id, principal.org_role) == (
        "dev@example.com",
        "org_1",
        "admin",
    )


def test_an_expired_session_is_rejected(keypair, settings):
    token = make_token(keypair, expires_in=-3600)
    with pytest.raises(AuthError) as excinfo:
        verify(token, settings)
    assert excinfo.value.status == 401
    assert "expired" in str(excinfo.value)


def test_a_token_from_another_issuer_is_rejected(keypair, settings):
    with pytest.raises(AuthError, match="not issued by"):
        verify(make_token(keypair, issuer="https://someone-else.clerk.accounts.dev"), settings)


def test_a_token_for_another_audience_is_rejected(keypair, jwks_json):
    settings = Settings(mode=Mode.ON, issuer=ISSUER, jwks_json=jwks_json, audience="https://api.me")
    with pytest.raises(AuthError, match="not addressed to this API"):
        verify(make_token(keypair, claims={"aud": "https://api.someone-else"}), settings)


def test_a_token_with_no_expiry_is_rejected(keypair, settings):
    now = int(time.time())
    token = jwt.encode(
        {"sub": "user_1", "iss": ISSUER, "iat": now, "nbf": now},
        keypair,
        algorithm="RS256",
        headers={"kid": KID},
    )
    with pytest.raises(AuthError):
        verify(token, settings)


def test_the_algorithm_may_not_be_negotiated_by_the_token(keypair, settings):
    """A token must not be able to pick HS256 and have the public key used as a secret.

    PyJWT refuses to *build* this forgery, so it is assembled by hand: the
    classic algorithm-confusion attack signs a token with HMAC keyed by the
    server's public key, and a verifier that trusts the header accepts it.
    """
    public_pem = keypair.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    def enc(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    now = int(time.time())
    signing_input = f"{enc({'alg': 'HS256', 'typ': 'JWT', 'kid': KID})}." + enc(
        {"sub": "user_admin", "iss": ISSUER, "iat": now, "exp": now + 60}
    )
    signature = hmac.new(public_pem, signing_input.encode(), hashlib.sha256).digest()
    forged = f"{signing_input}.{base64.urlsafe_b64encode(signature).decode().rstrip('=')}"

    with pytest.raises(AuthError, match="algorithm"):
        verify(forged, settings)


def test_a_tampered_payload_fails_the_signature(keypair, settings):
    token = make_token(keypair, subject="user_someone_else")
    header, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["sub"] = "user_admin"
    forged_payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    with pytest.raises(AuthError):
        verify(f"{header}.{forged_payload}.{signature}", settings)


def test_an_unknown_key_id_is_rejected(keypair, settings):
    with pytest.raises(AuthError, match="unknown key id"):
        verify(make_token(keypair, kid="ins_notpublished"), settings)


def test_an_unauthorized_party_is_rejected(keypair, jwks_json):
    settings = Settings(
        mode=Mode.ON,
        issuer=ISSUER,
        jwks_json=jwks_json,
        authorized_parties=("https://app.example.com",),
    )
    assert verify(make_token(keypair), settings).user_id == "user_2xYz"
    with pytest.raises(AuthError, match="not an authorized party"):
        verify(make_token(keypair, claims={"azp": "https://attacker.example"}), settings)


def test_verification_refuses_to_run_when_auth_is_off(keypair):
    with pytest.raises(AuthError) as excinfo:
        verify(make_token(keypair), Settings(mode=Mode.OFF))
    assert excinfo.value.status == 503


def test_a_malformed_jwks_is_a_startup_error():
    settings = Settings(mode=Mode.ON, issuer=ISSUER, jwks_json='{"keys": "nope"}')
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "RS256", "kid": "kid1"}).encode()
    ).decode().rstrip("=")
    with pytest.raises(AuthConfigError):
        verify(f"{header}.eyJzdWIiOiJ4In0.c2ln", settings)


# ── Middleware ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    @app.get("/repos")
    def repos(request: Request) -> dict:
        user = getattr(request.state, "user", None)
        return {"user": user.user_id if user else None}

    app.add_middleware(ClerkAuthMiddleware, settings=settings)
    return TestClient(app)


def test_public_paths_stay_public(client):
    assert client.get("/health").status_code == 200


def test_the_spa_shell_loads_before_sign_in(client):
    assert client.get("/").status_code == 404  # no route, but not a 401
    assert is_public("GET", "/assets/index-abc123.js")
    assert is_public("GET", "/index.html")


def test_a_data_route_without_a_session_is_401(client):
    response = client.get("/repos")
    assert response.status_code == 401
    assert response.json()["detail"] == "sign in required"
    assert response.headers["www-authenticate"] == 'Bearer realm="gatekeeper"'


def test_a_data_route_with_a_session_reports_the_user(client, keypair):
    response = client.get("/repos", headers={"Authorization": f"Bearer {make_token(keypair)}"})
    assert response.status_code == 200
    assert response.json()["user"] == "user_2xYz"


def test_a_non_bearer_scheme_is_401(client):
    response = client.get("/repos", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401


def test_an_invalid_token_is_401_not_500(client, keypair):
    response = client.get("/repos", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_preflight_is_not_challenged(client):
    response = client.options("/repos", headers={"Origin": "http://localhost:3000"})
    assert response.status_code != 401


def test_off_mode_is_a_pass_through(keypair):
    app = FastAPI()

    @app.get("/repos")
    def repos(request: Request) -> dict:
        return {"user": getattr(getattr(request, "state", None), "user", None) and "set"}

    app.add_middleware(ClerkAuthMiddleware, settings=Settings(mode=Mode.OFF))
    assert TestClient(app).get("/repos").status_code == 200
