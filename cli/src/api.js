/**
 * HTTP client for the Gatekeeper API.
 *
 * Two rules keep the pre-push hook trustworthy:
 *
 * 1. It fails open. An unreachable API warns and returns null, because a
 *    network blip must never block a push -- only a verdict from the API can.
 * 2. It sends the session token when there is one, so a deployment with Clerk
 *    enabled works without the CLI knowing anything about Clerk.
 *
 * The token comes from `GATEKEEPER_TOKEN` (CI) or `~/.gatekeeper/config.json`
 * (a developer who ran `gatekeeper auth --token ...`).
 */

import { loadUserConfig } from './config.js'

export const DEFAULT_TIMEOUT_MS = 10000

export function resolveApiUrl(userConfig = loadUserConfig(), env = process.env) {
  const url = env.GATEKEEPER_API_URL || userConfig.api_url || ''
  return url.replace(/\/+$/, '')
}

export function resolveToken(userConfig = loadUserConfig(), env = process.env) {
  return env.GATEKEEPER_TOKEN || userConfig.token || ''
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

/**
 * One request. Throws ApiError on an HTTP error, returns null when the API
 * cannot be reached at all -- callers decide which of those is fatal.
 */
export async function request(path, { method = 'GET', body, userConfig, env, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const base = resolveApiUrl(userConfig, env)
  if (!base) throw new ApiError(0, 'no API URL configured')

  const headers = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  const token = resolveToken(userConfig, env)
  if (token) headers.Authorization = `Bearer ${token}`

  let response
  try {
    response = await fetch(`${base}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    throw new ApiError(0, `API unreachable (${error.message})`)
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const parsed = await response.json()
      if (parsed?.detail) detail = parsed.detail
    } catch {
      // Keep the status line.
    }
    throw new ApiError(response.status, detail)
  }
  return response.json()
}

/** Score one commit's features. Returns null when the API is unavailable. */
export async function scoreCommit(features, repoName, options = {}) {
  try {
    return await request('/predict', {
      method: 'POST',
      body: { features, repo_name: repoName },
      ...options,
    })
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      throw error // a rejected token is worth stopping for; see commands.js
    }
    if (!options.quiet) {
      console.warn(`⚠  ${error.message}. Scoring locally only.`)
    }
    return null
  }
}
