/**
 * Sign-in gate for the dashboard.
 *
 * The API is the authority on whether auth is on -- it reports that at
 * `/auth/config`, which is public -- so the frontend never decides by itself
 * whether to trust a session. That also means a deployment can switch Clerk on
 * without rebuilding the bundle, and `npm run dev` keeps working with no keys.
 *
 * When auth is off the app renders exactly as before. When it is on, nothing
 * but the sign-in card is rendered until Clerk reports a session, and every
 * request carries that session's token (see api.js).
 */

import { useEffect, useState } from 'react'
import { ClerkProvider, SignIn, UserButton, useAuth } from '@clerk/react'
import App from './App.jsx'
import { authConfig, setTokenProvider } from './api.js'

function SessionBridge({ children }) {
  const { isLoaded, isSignedIn, getToken } = useAuth()

  useEffect(() => {
    setTokenProvider(async () => (isSignedIn ? getToken() : null))
    return () => setTokenProvider(async () => null)
  }, [isSignedIn, getToken])

  if (!isLoaded) return <div className="loading">Checking session…</div>
  if (!isSignedIn) {
    return (
      <div className="signin-shell">
        <div className="signin-card">
          <h1>Gatekeeper</h1>
          <p>ML risk scores for every commit. Sign in to see your repositories.</p>
          <SignIn routing="hash" />
        </div>
      </div>
    )
  }
  return children
}

function AccountChip() {
  const { isSignedIn } = useAuth()
  return isSignedIn ? <UserButton afterSignOutUrl="/" /> : null
}

export default function ClerkRoot() {
  const [config, setConfig] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    authConfig().then(setConfig).catch((e) => setError(e.message))
  }, [])

  if (error) {
    // Fail visibly rather than showing a dashboard whose requests will 401.
    return (
      <div className="signin-shell">
        <div className="signin-card">
          <h1>Cannot reach the API</h1>
          <p>{error}</p>
        </div>
      </div>
    )
  }
  if (!config) return <div className="loading">Loading…</div>
  if (!config.enabled) return <App />
  if (!config.publishable_key) {
    return (
      <div className="signin-shell">
        <div className="signin-card">
          <h1>Sign-in is misconfigured</h1>
          <p>
            The API requires a session but did not publish a Clerk publishable key. Set
            <code> CLERK_PUBLISHABLE_KEY </code> on the server.
          </p>
        </div>
      </div>
    )
  }

  return (
    <ClerkProvider publishableKey={config.publishable_key} afterSignOutUrl="/">
      <SessionBridge>
        <App accountSlot={<AccountChip />} />
      </SessionBridge>
    </ClerkProvider>
  )
}
