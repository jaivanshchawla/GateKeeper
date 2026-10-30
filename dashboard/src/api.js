/**
 * One place that talks to the API.
 *
 * Every request needs the same two things: the configured base URL and -- once
 * Clerk is enabled -- the current session token. Keeping the token provider
 * here means no component has to remember to attach it, and the signed-out
 * branch is a single lookup rather than a prop threaded through the tree.
 */

export const API = import.meta.env.VITE_API_URL || ''

let tokenProvider = async () => null

/** Registered by the Clerk bridge; a no-op when auth is disabled. */
export function setTokenProvider(provider) {
  tokenProvider = provider
}

async function headers(extra) {
  const token = await tokenProvider()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra }
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

async function request(path, init) {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body?.detail) detail = body.detail
    } catch {
      // A non-JSON error body is still an error; keep the status line.
    }
    throw new ApiError(response.status, detail)
  }
  return response.json()
}

export function api(path) {
  return headers({}).then((h) => request(path, { headers: h }))
}

export function apiPost(path, body) {
  return headers({ 'Content-Type': 'application/json' }).then((h) =>
    request(path, { method: 'POST', headers: h, body: JSON.stringify(body) }),
  )
}

/** Public: the dashboard asks this before it has a session. */
export async function authConfig() {
  const response = await fetch(`${API}/auth/config`)
  if (!response.ok) throw new ApiError(response.status, 'auth config unavailable')
  return response.json()
}
