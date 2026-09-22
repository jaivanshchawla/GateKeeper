#!/usr/bin/env node
/**
 * Capture screenshots of all 7 dashboard views, light and dark, over the
 * Chrome DevTools Protocol. Zero npm dependencies (Node >= 22 for global WebSocket).
 *
 * Usage: node scripts/capture_screenshots.mjs [baseUrl] [outDir]
 * Defaults: http://127.0.0.1:8000  docs/screenshots/
 *
 * Waits for a known data element per view before capturing — an
 * initial-render capture would photograph the empty state.
 */
import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

const BASE = process.argv[2] || 'http://127.0.0.1:8000'
const OUT = process.argv[3] || 'docs/screenshots'
const PORT = 9333
const CHROME = process.env.CHROME_PATH ||
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

mkdirSync(OUT, { recursive: true })

// ── launch chrome ────────────────────────────────────────────
if (!existsSync(CHROME)) {
  console.error(`Chrome not found at ${CHROME}; set CHROME_PATH`)
  process.exit(1)
}
const chrome = spawn(CHROME, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  '--no-first-run', '--no-default-browser-check',
  '--disable-extensions',
  `--user-data-dir=${process.env.TMPDIR || '/tmp'}/gk-shot-profile`,
  '--window-size=1280,900',
  'about:blank',
], { stdio: 'ignore' })

const sleep = ms => new Promise(r => setTimeout(r, ms))

async function findTarget() {
  for (let i = 0; i < 40; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/list`)
      const targets = await r.json()
      const page = targets.find(t => t.type === 'page')
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl
    } catch {}
    await sleep(250)
  }
  throw new Error('Chrome DevTools endpoint never came up')
}

// ── minimal CDP client ───────────────────────────────────────
const ws = new WebSocket(await findTarget())
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej })

let nextId = 1
const pending = new Map()
ws.onmessage = ev => {
  const msg = JSON.parse(ev.data)
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id)
    pending.delete(msg.id)
    msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result)
  }
}
function cdp(method, params = {}) {
  const id = nextId++
  ws.send(JSON.stringify({ id, method, params }))
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
}

async function evaluate(expression) {
  const r = await cdp('Runtime.evaluate', {
    expression, awaitPromise: true, returnByValue: true,
  })
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + ' ' +
    (r.exceptionDetails.exception?.description || ''))
  return r.result.value
}

async function waitFor(expression, label, timeoutMs = 20000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    try { if (await evaluate(expression)) return } catch {}
    await sleep(300)
  }
  throw new Error(`Timed out waiting for: ${label}`)
}

async function setTheme(theme) {
  await evaluate(`localStorage.setItem('gk-theme', '${theme}'); 'ok'`)
}

async function goto(path = '/') {
  await cdp('Page.navigate', { url: BASE + path })
  await waitFor("document.readyState === 'complete'", 'document load')
}

async function capture(filename) {
  // full-page clip so table-heavy views aren't cut off
  const height = await evaluate(
    "Math.min(Math.max(document.documentElement.scrollHeight, 900), 12000)")
  const r = await cdp('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: 1280, height, scale: 1 },
  })
  writeFileSync(join(OUT, filename), Buffer.from(r.data, 'base64'))
  const kb = Math.round(r.data.length * 0.75 / 1024)
  console.log(`  saved ${filename} (${kb} KB, ${Math.round(height)}px tall)`)
}

// ── view definitions: nav = expression evaluated in page, ready = data predicate ──
const NAV = "(label) => [...document.querySelectorAll('.top-nav button')]" +
  ".find(b => b.textContent === label)?.click()"

const VIEWS = [
  {
    name: 'overview',
    nav: "'no-op'",                       // default view, no nav needed
    ready: "document.querySelectorAll('.repo-card').length >= 5",
    readyLabel: 'repo cards',
  },
  {
    name: 'repo-detail',
    nav: `(${NAV})('Repo Detail')`,
    ready: "document.querySelectorAll('table tbody tr').length > 10",
    readyLabel: 'commit rows',
  },
  {
    name: 'commit-detail',
    // repo-detail → wait for rows to render → first "Detail" button
    nav: `(${NAV})('Repo Detail')`,
    waitBefore: "document.querySelectorAll('table tbody tr .toggle-btn').length > 0",
    waitBeforeLabel: 'repo rows for detail click',
    after: `() => { const b = [...document.querySelectorAll('table tbody tr .toggle-btn')][0]; b && b.click() }`,
    ready: "document.querySelector('h1')?.textContent?.startsWith('Commit')",
    readyLabel: 'commit header',
  },
  {
    name: 'prs',
    nav: `(${NAV})('PRs')`,
    ready: "document.querySelectorAll('table tbody tr').length > 0",
    readyLabel: 'PR rows',
  },
  {
    name: 'file-detail',
    nav: `(${NAV})('File Detail')`,
    after: `() => {
      const input = document.querySelector('input[placeholder="File path..."]')
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value').set
      setter.call(input, 'docs/releases/6.1.txt')
      input.dispatchEvent(new Event('input', { bubbles: true }))
      setTimeout(() => {
        const btn = [...document.querySelectorAll('button')]
          .find(b => b.textContent === 'Search'); btn && btn.click()
      }, 50)
    }`,
    ready: "document.querySelector('h2 code')?.textContent?.includes('docs/releases')",
    readyLabel: 'file history card',
  },
  {
    name: 'config-editor',
    nav: `(${NAV})('Config')`,
    ready: "document.querySelectorAll('table tbody tr').length >= 15",
    readyLabel: 'rule rows',
  },
  {
    name: 'model-health',
    nav: `(${NAV})('Model Health')`,
    ready: "document.querySelectorAll('.card').length > 0",
    readyLabel: 'health cards',
  },
]

// ── main: fresh navigation per shot so state can't leak ──────
await cdp('Page.enable')
await cdp('Runtime.enable')
await cdp('Emulation.setDeviceMetricsOverride', {
  width: 1280, height: 900, deviceScaleFactor: 1, mobile: false,
})

let failures = 0
for (const theme of ['light', 'dark']) {
  console.log(`── ${theme} ──`)
  for (const view of VIEWS) {
    try {
      await goto('/')
      await setTheme(theme)          // set before load is read…
      await goto('/')                // …so reload applies it
      const themeApplied = await evaluate(
        "document.documentElement.getAttribute('data-theme')")
      if (themeApplied !== theme) throw new Error(`theme not applied (got ${themeApplied})`)
      if (view.nav !== "'no-op'") { await evaluate(view.nav); await sleep(400) }
      if (view.waitBefore) await waitFor(view.waitBefore, view.waitBeforeLabel || 'precondition')
      if (view.after) { await evaluate(`(${view.after})()`); await sleep(400) }
      await waitFor(view.ready, view.readyLabel)
      await sleep(600)               // charts settle
      await capture(`${view.name}-${theme}.png`)
    } catch (e) {
      failures++
      console.error(`  FAIL ${view.name}-${theme}: ${e.message}`)
      // capture whatever state it's in for debugging
      try { await capture(`${view.name}-${theme}-FAILED.png`) } catch {}
    }
  }
}

ws.close()
chrome.kill()
console.log(failures === 0 ? 'ALL 14 CAPTURES OK' : `${failures} FAILURES`)
process.exit(failures === 0 ? 0 : 1)
