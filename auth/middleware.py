#!/usr/bin/env python3
"""Enforce Clerk sessions on the API, in one place rather than per route.

A dependency on each handler is only as good as the next handler someone
forgets to decorate, so the check is middleware: on by default, with an explicit
public list. With ``GATEKEEPER_AUTH_MODE=off`` the middleware is a pass-through
and ``request.state.user`` is simply absent, which is what keeps local
development and the existing tests working unchanged.

The dashboard shell has to load while signed out -- that is how a user reaches
the sign-in screen -- so static assets are public and only the data routes are
gated.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from anyio import to_thread

from .clerk import AuthError, Principal, Settings, verify

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/metrics",
        "/auth/config",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
)

# Extensions the built dashboard needs before anybody has signed in.
STATIC_SUFFIXES = (
    ".html", ".js", ".mjs", ".css", ".map", ".png", ".jpg", ".jpeg", ".svg",
    ".ico", ".webp", ".woff", ".woff2", ".ttf", ".json.txt",
)


def is_public(method: str, path: str) -> bool:
    """Whether a request may proceed without a session.

    Reads of the SPA shell and its assets are public; every API route is not.
    ``OPTIONS`` is public because the browser's preflight carries no
    ``Authorization`` header by design.
    """
    if method == "OPTIONS":
        return True
    if path in PUBLIC_PATHS or path.rstrip("/") == "":
        return True
    return path.lower().endswith(STATIC_SUFFIXES)


def _bearer(scope: Scope) -> str:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            raw = value.decode("latin-1").strip()
            scheme, _, token = raw.partition(" ")
            if scheme.lower() == "bearer":
                return token.strip()
            return ""
    return ""


async def _respond(send: Send, status: int, detail: str, extra: list[tuple[bytes, bytes]]) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers + extra})
    await send({"type": "http.response.body", "body": body})


class ClerkAuthMiddleware:
    """Authenticate every non-public request, or reject it."""

    def __init__(self, app: Any, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.settings.enabled:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        if is_public(method, path):
            await self.app(scope, receive, send)
            return

        token = _bearer(scope)
        if not token:
            await _respond(
                send,
                401,
                "sign in required",
                [(b"www-authenticate", b'Bearer realm="gatekeeper"')],
            )
            return

        try:
            # Verification may fetch the JWKS on a cache miss, and blocking the
            # event loop for that would stall every other request.
            principal: Principal = await to_thread.run_sync(verify, token, self.settings)
        except AuthError as exc:
            extra = (
                [(b"www-authenticate", b'Bearer realm="gatekeeper"')]
                if exc.status == 401
                else []
            )
            await _respond(send, exc.status, exc.message, extra)
            return

        scope.setdefault("state", {})["user"] = principal
        await self.app(scope, receive, send)


__all__ = ["PUBLIC_PATHS", "ClerkAuthMiddleware", "is_public"]
