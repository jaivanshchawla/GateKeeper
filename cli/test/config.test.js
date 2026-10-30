import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  DEFAULT_CONFIG,
  evaluateRules,
  isTestFile,
  mergeConfig,
} from '../src/config.js'

const diff = (overrides = {}) => ({
  files: [],
  linesAdded: 0,
  linesDeleted: 0,
  date: '2026-03-04T10:00:00Z', // a Wednesday
  ...overrides,
})

test('repo rules override defaults one key at a time', () => {
  const merged = mergeConfig({ rules: { large_change: { max_lines: 100 } } })
  assert.equal(merged.rules.large_change.max_lines, 100)
  assert.equal(merged.rules.large_change.severity, 'warn')
  assert.deepEqual(merged.rules.too_many_files, DEFAULT_CONFIG.rules.too_many_files)
})

test('a missing repo config is the defaults', () => {
  assert.deepEqual(mergeConfig(null), DEFAULT_CONFIG)
})

test('large_change fires above the limit and not at it', () => {
  const files = [{ path: 'src/app.py', added: 300, deleted: 200 }]
  const atLimit = evaluateRules(diff({ files, linesAdded: 300, linesDeleted: 200 }), DEFAULT_CONFIG)
  assert.equal(atLimit.filter((v) => v.rule === 'large_change').length, 0)

  const over = evaluateRules(diff({ files, linesAdded: 301, linesDeleted: 200 }), DEFAULT_CONFIG)
  const largeChange = over.filter((v) => v.rule === 'large_change')
  assert.equal(largeChange.length, 1)
  assert.equal(largeChange[0].severity, 'warn')
})

test('too_many_files counts files, not lines', () => {
  const files = Array.from({ length: 21 }, (_, i) => ({ path: `src/f${i}.py`, added: 1, deleted: 0 }))
  const violations = evaluateRules(diff({ files, linesAdded: 21 }), DEFAULT_CONFIG)
  assert.ok(violations.some((v) => v.rule === 'too_many_files'))
})

test('no_tests is satisfied by any test file in the change', () => {
  const files = [
    { path: 'src/app.py', added: 5, deleted: 0 },
    { path: 'tests/test_app.py', added: 5, deleted: 0 },
  ]
  const violations = evaluateRules(diff({ files, linesAdded: 10 }), DEFAULT_CONFIG)
  assert.equal(violations.filter((v) => v.rule === 'no_tests').length, 0)
})

test('no_tests is not raised for a docs-only change', () => {
  const files = [{ path: 'docs/usage.md', added: 5, deleted: 0 }]
  const violations = evaluateRules(diff({ files, linesAdded: 5 }), DEFAULT_CONFIG)
  assert.equal(violations.length, 0)
})

test('config_and_code fires only when both kinds are touched', () => {
  const both = [
    { path: '.github/workflows/ci.yml', added: 1, deleted: 0 },
    { path: 'src/main.py', added: 1, deleted: 0 },
  ]
  assert.ok(
    evaluateRules(diff({ files: both, linesAdded: 2 }), DEFAULT_CONFIG).some(
      (v) => v.rule === 'config_and_code',
    ),
  )
  const onlyConfig = [{ path: 'docker-compose.yml', added: 1, deleted: 0 }]
  assert.equal(
    evaluateRules(diff({ files: onlyConfig, linesAdded: 1 }), DEFAULT_CONFIG).filter(
      (v) => v.rule === 'config_and_code',
    ).length,
    0,
  )
})

test('weekend_deploy reads the commit date', () => {
  const saturday = diff({ date: '2026-03-07T10:00:00Z' })
  assert.ok(evaluateRules(saturday, DEFAULT_CONFIG).some((v) => v.rule === 'weekend_deploy'))
})

test('a disabled rule stays silent', () => {
  const config = mergeConfig({ rules: { large_change: { enabled: false } } })
  const files = [{ path: 'src/app.py', added: 900, deleted: 0 }]
  const violations = evaluateRules(diff({ files, linesAdded: 900 }), config)
  assert.equal(violations.filter((v) => v.rule === 'large_change').length, 0)
})

test('test files are recognised in the layouts used by the corpus', () => {
  for (const path of ['tests/test_api.py', 'src/app.test.js', 'pkg/x_spec.rb', '__tests__/a.js']) {
    assert.equal(isTestFile(path), true, path)
  }
  for (const path of ['src/app.py', 'docs/testing.md', 'contest/results.py']) {
    assert.equal(isTestFile(path), false, path)
  }
})
