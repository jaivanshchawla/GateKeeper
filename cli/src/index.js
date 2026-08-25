#!/usr/bin/env node
/**
 * Gatekeeper CLI — ML-powered commit risk scoring with deterministic rules.
 *
 * Commands:
 *   gatekeeper init       Install pre-push hook, write .gatekeeper.yml
 *   gatekeeper check      Score unpushed commits locally
 *   gatekeeper status     Per-repo gate status from dashboard
 *   gatekeeper explain <sha>  Full breakdown for one commit
 *   gatekeeper config     Show effective config
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, basename } from 'path';
import { homedir } from 'os';
import { parse as parseYaml } from 'yaml';

// ── Helpers ──────────────────────────────────────────────────────────

function getGitRoot() {
  try {
    return execSync('git rev-parse --show-toplevel', { encoding: 'utf-8' }).trim();
  } catch {
    return null;
  }
}

function getGitRemoteUrl() {
  try {
    return execSync('git remote get-url origin', { encoding: 'utf-8' }).trim();
  } catch {
    return null;
  }
}

function getUnpushedCommits() {
  try {
    // Use git rev-list to avoid shell escaping issues with ^
    // Get remote branch name
    let remoteBranch = '';
    try {
      remoteBranch = execSync('git rev-parse --abbrev-ref @{u} 2>/dev/null', { encoding: 'utf-8' }).trim();
    } catch {
      try {
        const branch = execSync('git branch --show-current', { encoding: 'utf-8' }).trim();
        remoteBranch = `origin/${branch}`;
      } catch {}
    }
    if (!remoteBranch) {
      // Fallback: use origin/main or origin/master
      try {
        execSync('git rev-parse --verify origin/main 2>/dev/null', { encoding: 'utf-8' });
        remoteBranch = 'origin/main';
      } catch {
        try {
          execSync('git rev-parse --verify origin/master 2>/dev/null', { encoding: 'utf-8' });
          remoteBranch = 'origin/master';
        } catch {}
      }
    }
    const hashes = remoteBranch
      ? execSync(`git rev-list --oneline ${remoteBranch}..HEAD`, { encoding: 'utf-8' }).trim()
      : '';
    if (!hashes) return [];
    return hashes.split('\n').filter(Boolean).map(hash => {
      const msg = execSync(`git log -1 --format=%s ${hash}`, { encoding: 'utf-8' }).trim();
      return { sha: hash, message: msg };
    });
  } catch {
    return [];
  }
}

function loadUserConfig() {
  const configPath = join(homedir(), '.gatekeeper', 'config.json');
  if (existsSync(configPath)) {
    return JSON.parse(readFileSync(configPath, 'utf-8'));
  }
  return {};
}

function saveUserConfig(config) {
  const dir = join(homedir(), '.gatekeeper');
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'config.json'), JSON.stringify(config, null, 2));
}

function loadRepoConfig(gitRoot) {
  const configPath = join(gitRoot, '.gatekeeper.yml');
  if (existsSync(configPath)) {
    return parseYaml(readFileSync(configPath, 'utf-8'));
  }
  return null;
}

const DEFAULT_CONFIG = {
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
};

function getEffectiveConfig(gitRoot) {
  const repoConfig = loadRepoConfig(gitRoot);
  if (!repoConfig) return DEFAULT_CONFIG;
  // Deep merge
  const merged = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
  if (repoConfig.rules) {
    for (const [name, cfg] of Object.entries(repoConfig.rules)) {
      merged.rules[name] = { ...(merged.rules[name] || {}), ...cfg };
    }
  }
  for (const key of ['ml_scoring', 'fail_on']) {
    if (repoConfig[key]) merged[key] = repoConfig[key];
  }
  return merged;
}

// ── API Client ───────────────────────────────────────────────────────

async function callApi(endpoint, body, userConfig) {
  const apiUrl = userConfig.api_url || process.env.GATEKEEPER_API_URL;
  if (!apiUrl) {
    console.warn('⚠  No API URL configured. Set GATEKEEPER_API_URL or run:');
    console.warn('   gatekeeper config --api-url https://your-api.render.com');
    return null;
  }
  try {
    const response = await fetch(`${apiUrl}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) {
      console.warn(`⚠  API returned ${response.status}: ${response.statusText}`);
      return null;
    }
    return await response.json();
  } catch (err) {
    console.warn(`⚠  API unreachable (${err.message}). Failing open — no block.`);
    return null;
  }
}

// ── Commands ─────────────────────────────────────────────────────────

function cmdInit() {
  const gitRoot = getGitRoot();
  if (!gitRoot) {
    console.error('❌ Not inside a git repository.');
    process.exit(1);
  }

  // 1. Write .gatekeeper.yml if not present
  const configPath = join(gitRoot, '.gatekeeper.yml');
  if (!existsSync(configPath)) {
    const defaultYaml = `# Gatekeeper configuration
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
`;
    writeFileSync(configPath, defaultYaml);
    console.log('✅ Created .gatekeeper.yml');
  } else {
    console.log('ℹ  .gatekeeper.yml already exists, skipping.');
  }

  // 2. Install pre-push hook
  const hooksDir = join(gitRoot, '.git', 'hooks');
  const hookPath = join(hooksDir, 'pre-push');
  const hookContent = `#!/bin/sh
# Gatekeeper pre-push hook
# Scores outgoing commits and warns on high risk.
# Skip with: SKIP=gatekeeper-score git push

if [ "$SKIP" = "gatekeeper-score" ] || [ "$SKIP" = "all" ]; then
  exit 0
fi

echo "🛡️  Gatekeeper: Scoring outgoing commits..."
npx gatekeeper check
exit $?
`;
  writeFileSync(hookPath, hookContent);
  // Make executable (Unix only; no-op on Windows)
  try { execSync(`chmod +x "${hookPath}"`); } catch {}
  console.log('✅ Installed pre-push hook');

  // 3. Register repo with dashboard (local user config)
  const remoteUrl = getGitRemoteUrl();
  const userConfig = loadUserConfig();
  if (remoteUrl) {
    const repoName = remoteUrl.replace(/^.*github\.com[:/]/, '').replace(/\.git$/, '');
    if (!userConfig.repos) userConfig.repos = {};
    userConfig.repos[repoName] = {
      remote_url: remoteUrl,
      registered_at: new Date().toISOString(),
    };
    saveUserConfig(userConfig);
    console.log(`✅ Registered repo: ${repoName}`);
  }

  console.log('\n🎯 Gatekeeper initialized. Run `gatekeeper check` to score unpushed commits.');
}

function cmdCheck() {
  const gitRoot = getGitRoot();
  if (!gitRoot) {
    console.error('❌ Not inside a git repository.');
    process.exit(1);
  }

  const commits = getUnpushedCommits();
  if (commits.length === 0) {
    console.log('✅ No unpushed commits to score.');
    return;
  }

  const config = getEffectiveConfig(gitRoot);
  console.log(`\n🛡️  Gatekeeper: Scoring ${commits.length} unpushed commit(s)...\n`);

  let hasBlock = false;

  for (const { sha, message } of commits) {
    // Simple local scoring — use git log to extract basic features
    let filesChanged = [];
    let linesAdded = 0;
    let linesDeleted = 0;
    try {
      const stats = execSync(`git diff-tree --no-commit-id --numstat ${sha}`, { encoding: 'utf-8' });
      for (const line of stats.trim().split('\n')) {
        const parts = line.split('\t');
        if (parts.length >= 3) {
          filesChanged.push(parts[2]);
          linesAdded += parseInt(parts[0]) || 0;
          linesDeleted += parseInt(parts[1]) || 0;
        }
      }
    } catch {}

    // Local rule checks (deterministic, no API needed)
    const violations = [];
    const totalLines = linesAdded + linesDeleted;

    if (totalLines > (config.rules.large_change?.max_lines || 500)) {
      violations.push({ rule: 'large_change', severity: 'warn', message: `${totalLines} lines changed` });
    }
    if (filesChanged.length > (config.rules.too_many_files?.max_files || 20)) {
      violations.push({ rule: 'too_many_files', severity: 'warn', message: `${filesChanged.length} files touched` });
    }

    const hasTests = filesChanged.some(f => /test|spec/i.test(f));
    if (!hasTests && (linesAdded > 0 || linesDeleted > 0)) {
      violations.push({ rule: 'no_tests', severity: 'warn', message: 'No test files included' });
    }

    const configPatterns = /\.ya?ml$|\.toml$|\.lock|dockerfile|\.github\/|docker-compose|makefile|\.env/i;
    const codePatterns = /\.(py|js|ts|jsx|tsx|rs|go|java|c|cpp|h)$/;
    const hasConfig = filesChanged.some(f => configPatterns.test(f));
    const hasCode = filesChanged.some(f => codePatterns.test(f));
    if (hasConfig && hasCode) {
      violations.push({ rule: 'config_and_code', severity: 'warn', message: 'Touches both config/CI and source code' });
    }

    // Render result
    const shortSha = sha.slice(0, 8);
    const shortMsg = message.slice(0, 60);

    if (violations.length === 0) {
      console.log(`  ✅ ${shortSha} ${shortMsg}`);
    } else {
      const hasBlocker = violations.some(v => v.severity === 'block');
      if (hasBlocker) hasBlock = true;

      const icon = hasBlocker ? '🚫' : '⚠️ ';
      console.log(`  ${icon} ${shortSha} ${shortMsg}`);
      for (const v of violations) {
        const sev = v.severity === 'block' ? 'BLOCK' : v.severity.toUpperCase();
        console.log(`     [${sev}] ${v.rule}: ${v.message}`);
      }
    }
  }

  console.log('');

  if (hasBlock && config.fail_on?.includes('block')) {
    console.log('🚫 Push blocked by Gatekeeper (block-severity rule triggered).');
    console.log('   Fix the issues above, or skip with: SKIP=gatekeeper-score git push');
    process.exit(1);
  }

  const warnings = commits.length; // simplified
  console.log(`Done. ${commits.length} commit(s) scored.`);
}

async function cmdStatus() {
  const remoteUrl = getGitRemoteUrl();
  if (!remoteUrl) {
    console.error('❌ No git remote configured.');
    process.exit(1);
  }

  const repoName = remoteUrl.replace(/^.*github\.com[:/]/, '').replace(/\.git$/, '');
  const userConfig = loadUserConfig();

  if (!userConfig.api_url) {
    console.log(`📦 Repo: ${repoName}`);
    console.log('ℹ  No API URL configured. Set it with:');
    console.log('   gatekeeper config --api-url https://your-api.render.com');
    return;
  }

  console.log(`🛡️  Gatekeeper status for ${repoName}...\n`);
  const data = await callApi(`/issues?repo=${encodeURIComponent(repoName)}`, null, userConfig);
  if (!data) return;

  const issues = Array.isArray(data) ? data : data.issues || [];
  const open = issues.filter(i => i.status === 'open');
  const resolved = issues.filter(i => i.status === 'resolved');

  console.log(`  Open issues:     ${open.length}`);
  console.log(`  Resolved issues: ${resolved.length}`);
  if (open.length > 0) {
    console.log('\n  Recent open issues:');
    for (const issue of open.slice(-5)) {
      console.log(`    [Gate ${issue.gate}] ${issue.type}: ${issue.details?.slice(0, 80) || 'no details'}`);
    }
  }
}

async function cmdExplain(sha) {
  if (!sha) {
    console.error('Usage: gatekeeper explain <commit-sha>');
    process.exit(1);
  }

  const gitRoot = getGitRoot();
  if (!gitRoot) {
    console.error('❌ Not inside a git repository.');
    process.exit(1);
  }

  const userConfig = loadUserConfig();
  const remoteUrl = getGitRemoteUrl();
  const repoName = remoteUrl?.replace(/^.*github\.com[:/]/, '').replace(/\.git$/, '') || 'unknown';

  console.log(`\n🛡️  Gatekeeper explain for ${sha.slice(0, 12)}...\n`);

  // Call API for full breakdown
  const result = await callApi('/explain', {
    commit_hash: sha,
    repo: repoName,
    repo_path: gitRoot,
  }, userConfig);

  if (result) {
    console.log(`  Band:    ${result.risk_label?.toUpperCase() || 'unknown'}`);
    console.log(`  Score:   ${result.risk_score?.toFixed(4) || 'N/A'}`);
    if (result.explanations?.length) {
      console.log('\n  Top factors:');
      for (const exp of result.explanations) {
        console.log(`    • ${exp.human_readable || exp.description}`);
      }
    }
    if (result.rule_results?.length) {
      console.log('\n  Rules:');
      for (const r of result.rule_results) {
        const icon = r.passed ? '✅' : (r.severity === 'block' ? '🚫' : '⚠️ ');
        console.log(`    ${icon} [${r.severity}] ${r.rule}: ${r.message}`);
      }
    }
  } else {
    // Offline fallback: show git log info
    try {
      const log = execSync(`git log -1 --format=%H%n%an%n%ai%n%s ${sha}`, { encoding: 'utf-8' });
      const [hash, author, date, subject] = log.trim().split('\n');
      console.log(`  Hash:    ${hash?.slice(0, 12)}`);
      console.log(`  Author:  ${author}`);
      console.log(`  Date:    ${date}`);
      console.log(`  Message: ${subject}`);

      const stats = execSync(`git diff-tree --no-commit-id --stat ${sha}`, { encoding: 'utf-8' });
      console.log(`\n  Changes:\n${stats}`);
    } catch (err) {
      console.error(`  ❌ Could not read commit: ${err.message}`);
    }
  }
}

function cmdConfig(args) {
  const gitRoot = getGitRoot();
  const userConfig = loadUserConfig();

  if (args.includes('--api-url')) {
    const idx = args.indexOf('--api-url');
    const url = args[idx + 1];
    if (!url) {
      console.error('Usage: gatekeeper config --api-url <url>');
      process.exit(1);
    }
    userConfig.api_url = url;
    saveUserConfig(userConfig);
    console.log(`✅ API URL set to: ${url}`);
    return;
  }

  // Show effective config
  console.log('\n📋 Effective Gatekeeper configuration:\n');
  if (gitRoot) {
    const repoConfig = loadRepoConfig(gitRoot);
    if (repoConfig) {
      console.log(`  Repo config: ${join(gitRoot, '.gatekeeper.yml')}`);
    } else {
      console.log('  Repo config: (none — using defaults)');
    }
  }

  const effective = gitRoot ? getEffectiveConfig(gitRoot) : DEFAULT_CONFIG;
  console.log(`  API URL: ${userConfig.api_url || '(not set)'}`);
  console.log(`  ML scoring: ${effective.ml_scoring?.enabled ? 'enabled' : 'disabled'}`);
  console.log(`  Fail on: ${effective.fail_on?.join(', ') || 'block'}`);
  console.log('\n  Rules:');
  for (const [name, cfg] of Object.entries(effective.rules)) {
    const sev = cfg.severity || 'warn';
    const extra = cfg.max_lines ? ` (max: ${cfg.max_lines})` : cfg.max_files ? ` (max: ${cfg.max_files})` : '';
    console.log(`    ${name}: ${sev}${extra}`);
  }
}

// ── Main ─────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const command = args[0];

switch (command) {
  case 'init':
    cmdInit();
    break;
  case 'check':
    cmdCheck();
    break;
  case 'status':
    await cmdStatus();
    break;
  case 'explain':
    await cmdExplain(args[1]);
    break;
  case 'config':
    cmdConfig(args.slice(1));
    break;
  default:
    console.log(`
🛡️  Gatekeeper CLI — ML-powered commit risk scoring

Usage:
  gatekeeper init           Install pre-push hook + .gatekeeper.yml
  gatekeeper check          Score unpushed commits with rules
  gatekeeper status         Per-repo status from dashboard
  gatekeeper explain <sha>  Full breakdown for one commit
  gatekeeper config         Show effective config

Environment:
  GATEKEEPER_API_URL        API endpoint for ML scoring (optional)

Examples:
  npx gatekeeper init
  npx gatekeeper check
  npx gatekeeper config --api-url https://your-api.render.com
`);
    break;
}
