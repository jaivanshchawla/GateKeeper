#!/usr/bin/env python3
"""The two endpoints a Clerk-based SPA needs from us.

``/auth/config`` is public because the dashboard must know whether to render a
sign-in screen before it has a session, and because reading the publishable key
at runtime means a deployment does not rebuild the frontend to change it.

``/auth/me`` is how the UI (and ``gatekeeper whoami``) confirms that the token
it holds is one this API accepts, and which user it belongs to.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from .clerk import Principal, Settings

router = APIRouter(prefix="/auth", tags=["auth"])


def _settings(request: Request) -> Settings:
    return request.app.state.auth_settings


@router.get("/config")
def auth_config(request: Request) -> dict:
    """What the browser needs to start Clerk, and nothing secret."""
    settings = _settings(request)
    return {
        "enabled": settings.enabled,
        "mode": settings.mode.value,
        "issuer": settings.issuer,
        "publishable_key": settings.publishable_key,
        "authorized_parties": list(settings.authorized_parties),
    }


@router.get("/me")
def auth_me(request: Request) -> dict:
    """The caller's identity, as proven by their session token."""
    settings = _settings(request)
    principal: Principal | None = getattr(request.state, "user", None)
    if principal is None:
        return {"authenticated": False, "mode": settings.mode.value, "user": None}
    return {"authenticated": True, "mode": settings.mode.value, "user": principal.to_dict()}


__all__ = ["router"]
