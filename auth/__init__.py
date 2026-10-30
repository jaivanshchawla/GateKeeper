#!/usr/bin/env python3
"""Clerk session authentication for the Gatekeeper API.

Import surface for the rest of the codebase::

    from auth import ClerkAuthMiddleware, Settings, configure

``configure(app)`` mounts the middleware and the ``/auth`` routes and stores the
resolved :class:`Settings` on ``app.state`` so handlers can read it. It is
called once from ``api.main``; everything else here is exercised directly by
``tests/test_auth.py``.
"""

from __future__ import annotations

import os
from typing import Any

from .clerk import (
    AuthConfigError,
    AuthError,
    Mode,
    Principal,
    Settings,
    clear_jwks_cache,
    issuer_from_publishable_key,
    verify,
)
from .middleware import PUBLIC_PATHS, ClerkAuthMiddleware, is_public
from .routes import router

ENV_CORS = "GATEKEEPER_CORS_ORIGINS"


def configure(app: Any, settings: Settings | None = None) -> Settings:
    """Attach authentication to ``app`` and return the settings in force."""
    resolved = settings or Settings.from_env()
    app.state.auth_settings = resolved
    app.add_middleware(ClerkAuthMiddleware, settings=resolved)

    # A dashboard served from another origin (Vite on :3000 in development, or
    # a static host) needs CORS. Empty by default: same-origin deployments are
    # the norm and a wildcard would let any site call this API with a user's
    # session token.
    origins = [o.strip() for o in os.environ.get(ENV_CORS, "").split(",") if o.strip()]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["authorization", "content-type"],
        )

    app.include_router(router)
    return resolved


__all__ = [
    "ENV_CORS",
    "PUBLIC_PATHS",
    "AuthConfigError",
    "AuthError",
    "ClerkAuthMiddleware",
    "Mode",
    "Principal",
    "Settings",
    "clear_jwks_cache",
    "configure",
    "is_public",
    "issuer_from_publishable_key",
    "router",
    "verify",
]
