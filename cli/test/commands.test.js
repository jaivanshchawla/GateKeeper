import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, writeFileSync, rmSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { cmdCheck, cmdInit, hookScript } from '../src/commands.js'

// Every command here is given an empty user config and an environment with no
// API URL, so the tests never touch the developer's ~/.gatekeeper or network.
const offline = { env: {}, userConfig: {} }

function makeRepo() {
  const dir = mkdtempSync(join(tmpdir(), 'gatekeeper-cli-'))
  const git = (...args) =>
    execFileSync('git', args, { cwd: dir, encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'] })
  git('init', '-q', '-b', 'main')
  git('config', 'user.email', 'dev@example.com')
  git('config', 'user.name', 'Dev')
  return { dir, git }
}

function commit(repo, message, files) {
  for (const [path, content] of Object.entries(files)) {
    const full = join(repo.dir, path)
    mkdirSync(join(full, '..'), { recursive: true })
    writeFileSync(full, content)
  }
  repo.git('add', '.')
  repo.git('commit', '-q', '-m', message)
}

function capture() {
  const lines = []
  return { lines, out: (line = '') => lines.push(line), text: () => lines.join('\n') }
}

test('the generated hook resolves the CLI at run time and honours SKIP', () => {
  const script = hookScript()
  assert.match(script, /^#!\/bin\/sh/)
  assert.match(script, /SKIP.*gatekeeper-score/)
  assert.match(script, /command -v gatekeeper/)
  assert.match(script, /node_modules\/\.bin\/gatekeeper/)
  // An absolute path would break the moment the package is installed elsewhere.
  assert.doesNotMatch(script, /\/cli\/src\/index\.js/)
})

test('init writes the config and a hook, then reports what it did', () => {
  const { dir } = makeRepo()
  try {
    const { lines, out } = capture()
    assert.equal(cmdInit({ cwd: dir, out }), 0)
    assert.match(lines.join('\n'), /Created \.gatekeeper\.yml/)
    assert.match(lines.join('\n'), /Installed pre-push hook/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('check reports nothing to do when every commit is on the upstream', async () => {
  const repo = makeRepo()
  const { dir } = repo
  try {
    commit(repo, 'initial', { 'README.md': 'hello\n' })
    repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
    const { lines, out } = capture()
    assert.equal(await cmdCheck({ cwd: dir, out, ...offline }), 0)
    assert.match(lines.join('\n'), /No unpushed commits/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('check blocks an unpushed commit that trips a block-severity rule', async () => {
  const repo = makeRepo()
  const { dir } = repo
  try {
    commit(repo, 'initial', { 'README.md': 'hello\n' })
    repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
    writeFileSync(join(dir, '.gatekeeper.yml'), 'rules:\n  large_change:\n    max_lines: 3\n    severity: block\n')
    commit(repo, 'huge change', { 'src/app.py': 'line\n'.repeat(50) })

    const { lines, out } = capture()
    const code = await cmdCheck({ cwd: dir, out, ...offline })
    assert.equal(code, 1)
    assert.match(lines.join('\n'), /blocked/i)
    assert.match(lines.join('\n'), /large_change/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('check --json is machine-readable and carries the same verdict', async () => {
  const repo = makeRepo()
  const { dir } = repo
  try {
    commit(repo, 'initial', { 'README.md': 'hello\n' })
    repo.git('update-ref', 'refs/remotes/origin/main', 'HEAD')
    commit(repo, 'add endpoint', { 'src/api.py': 'print(1)\n' })

    const { lines, out } = capture()
    await cmdCheck({ cwd: dir, argv: ['--json'], out, ...offline })
    const payload = JSON.parse(lines.join('\n'))
    assert.equal(payload.summary.total, 1)
    assert.equal(payload.commits[0].band, 'medium') // no_tests warns
    assert.equal(payload.config.model_used, false)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
