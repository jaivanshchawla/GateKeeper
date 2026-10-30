/**
 * Configuration: defaults, the repo override, and the per-user settings file.
 *
 * Kept separate from the commands so the merge and the rule evaluation can be
 * tested without a git checkout or a network call.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { homedir } from 'os'
import { join } from 'path'
import { parse as parseYaml } from 'yaml'

export const DEFAULT_CONFIG = {
  rules: {
    large_change: { max_lines: 500, severity: 'warn' },
    too_many_files: { max_files: 20, severity: 'warn' },
    no_tests: { severity: 'warn', exempt_paths: ['docs/**', '*.md'] },
    config_and_code: { severity: 'warn' },
    revert_hotspot: { revert_count: 3, window_days: 60, severity: 'block' },
    first_touch: { severity: 'info' },
    weekend_deploy: { severity: 'info' },
    stale_file: { days: 180, severity: 'info' },
    direct_to_main: { severity: 'warn' },
  },
  ml_scoring: { enabled: true, band_thresholds: { high: 0.90, medium: 0.75 } },
  fail_on: ['block'],
}

export const DEFAULT_CONFIG_YAML = `# Gatekeeper configuration
# See https://github.com/jaivanshchawla/GateKeeper for options

rules:
  large_change:
    max_lines: 500
    severity: warn
  too_many_files:
    max_files: 20
    severity: warn
  no_tests:
    severity: warn
    exempt_paths:
      - "docs/**"
      - "*.md"
  config_and_code:
    severity: warn
  revert_hotspot:
    revert_count: 3
    window_days: 60
    severity: block
  first_touch:
    severity: info
  weekend_deploy:
    severity: info
  stale_file:
    days: 180
    severity: info
  direct_to_main:
    severity: warn

ml_scoring:
  enabled: true
  band_thresholds:
    high: 0.90
    medium: 0.75

fail_on:
  - block
`

export function userConfigPath() {
  return join(homedir(), '.gatekeeper', 'config.json')
}

export function loadUserConfig() {
  const path = userConfigPath()
  if (!existsSync(path)) return {}
  try {
    return JSON.parse(readFileSync(path, 'utf-8'))
  } catch {
    // A corrupt settings file must not stop a push; fall back to defaults.
    return {}
  }
}

export function saveUserConfig(config) {
  const dir = join(homedir(), '.gatekeeper')
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(userConfigPath(), JSON.stringify(config, null, 2))
}

export function loadRepoConfig(gitRoot) {
  if (!gitRoot) return null
  const path = join(gitRoot, '.gatekeeper.yml')
  if (!existsSync(path)) return null
  try {
    return parseYaml(readFileSync(path, 'utf-8'))
  } catch {
    return null
  }
}

/** Repo settings over the defaults, one rule at a time. */
export function mergeConfig(repoConfig) {
  const merged = JSON.parse(JSON.stringify(DEFAULT_CONFIG))
  if (!repoConfig) return merged
  if (repoConfig.rules) {
    for (const [name, cfg] of Object.entries(repoConfig.rules)) {
      merged.rules[name] = { ...(merged.rules[name] || {}), ...cfg }
    }
  }
  for (const key of ['ml_scoring', 'fail_on']) {
    if (repoConfig[key]) merged[key] = repoConfig[key]
  }
  return merged
}

export function getEffectiveConfig(gitRoot) {
  return mergeConfig(loadRepoConfig(gitRoot))
}

const TEST_FILE = /(^|\/)(tests?|__tests__|spec)\/|\.(test|spec)\.[a-z]+$|_test\.(py|go|rs)$|_spec\.(rb|py|js|ts|go|rs)$/i
const TEST_EXEMPT = [/^docs\//, /\.md$/]
const CONFIG_FILE = /\.ya?ml$|\.toml$|\.lock$|(^|\/)dockerfile$|\.github\/|docker-compose|makefile|\.env/i
const CODE_FILE = /\.(py|js|jsx|ts|tsx|rs|go|java|kt|c|cc|cpp|h|hpp|rb|php|cs|swift)$/i

export function isTestFile(path) {
  return TEST_FILE.test(path)
}

/**
 * Every deterministic rule, evaluated on one commit's diff.
 *
 * Pure on purpose: `check` is the product surface people run on every push, so
 * what it decides has to be testable without committing anything. Rules whose
 * inputs are not available locally (revert hotspots, stale files, first-touch)
 * are left to the API; inventing them from a single diff would be a guess.
 */
export function evaluateRules(diff, config) {
  const rules = config.rules || DEFAULT_CONFIG.rules
  const violations = []
  const totalLines = diff.linesAdded + diff.linesDeleted
  const files = diff.files || []
  const on = (rule) => rule && rule.enabled !== false

  const largeChange = on(rules.large_change) ? rules.large_change : null
  if (largeChange && totalLines > (largeChange.max_lines ?? 500)) {
    violations.push({
      rule: 'large_change',
      severity: largeChange.severity || 'warn',
      message: `${totalLines} lines changed (limit ${largeChange.max_lines ?? 500})`,
    })
  }

  const tooMany = on(rules.too_many_files) ? rules.too_many_files : null
  if (tooMany && files.length > (tooMany.max_files ?? 20)) {
    violations.push({
      rule: 'too_many_files',
      severity: tooMany.severity || 'warn',
      message: `${files.length} files touched (limit ${tooMany.max_files ?? 20})`,
    })
  }

  const noTests = on(rules.no_tests) ? rules.no_tests : null
  const changedPaths = files.map((f) => f.path)
  const onlyExempt = changedPaths.every((p) =>
    TEST_EXEMPT.some((pattern) => pattern.test(p)),
  )
  if (noTests && changedPaths.length > 0 && !onlyExempt && !changedPaths.some(isTestFile)) {
    violations.push({
      rule: 'no_tests',
      severity: noTests.severity || 'warn',
      message: 'No test file touched',
    })
  }

  if (
    on(rules.config_and_code) &&
    changedPaths.some((p) => CONFIG_FILE.test(p)) &&
    changedPaths.some((p) => CODE_FILE.test(p))
  ) {
    violations.push({
      rule: 'config_and_code',
      severity: rules.config_and_code.severity || 'warn',
      message: 'Touches both config/CI and source code',
    })
  }

  const weekend = on(rules.weekend_deploy) ? rules.weekend_deploy : null
  if (weekend && diff.date) {
    const day = new Date(diff.date).getUTCDay()
    if (day === 0 || day === 6) {
      violations.push({
        rule: 'weekend_deploy',
        severity: weekend.severity || 'info',
        message: 'Committed on a weekend',
      })
    }
  }

  return violations
}
