/**
 * Everything that shells out to git.
 *
 * `check` runs on every push, so these helpers are deliberately forgiving: a
 * missing remote or a detached HEAD degrades to an empty answer instead of
 * taking the push down with it.
 */

import { execSync } from 'child_process'

function git(args, { cwd, encoding = 'utf-8' } = {}) {
  return execSync(`git ${args}`, {
    cwd,
    encoding,
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim()
}

export function getGitRoot(cwd) {
  try {
    return git('rev-parse --show-toplevel', { cwd })
  } catch {
    return null
  }
}

export function getGitRemoteUrl(cwd) {
  try {
    return git('remote get-url origin', { cwd })
  } catch {
    return null
  }
}

/** `owner/name` from any GitHub remote form, or null when there is no remote. */
export function repoNameFromRemote(url) {
  if (!url) return null
  const match = url.match(/github\.com[:/]([^/]+\/[^/]+?)(?:\.git)?\/?$/)
  if (match) return match[1]
  // A non-GitHub remote still names the repository: use its last path segment.
  const last = url.replace(/\/$/, '').split(/[/:]/).pop()
  return last ? last.replace(/\.git$/, '') : null
}

export function getRepoName(cwd) {
  return repoNameFromRemote(getGitRemoteUrl(cwd))
}

export function getCurrentBranch(cwd) {
  try {
    return git('rev-parse --abbrev-ref HEAD', { cwd })
  } catch {
    return null
  }
}

/**
 * Commits on HEAD that the upstream branch does not have.
 *
 * Falls back to origin/main or origin/master when HEAD has no upstream, which
 * is the common case on a fresh feature branch.
 */
export function getUnpushedCommits(cwd) {
  const candidates = []
  try {
    candidates.push(git('rev-parse --abbrev-ref @{u}', { cwd }))
  } catch {
    const branch = getCurrentBranch(cwd)
    if (branch && branch !== 'HEAD') candidates.push(`origin/${branch}`)
    candidates.push('origin/main', 'origin/master')
  }

  for (const upstream of candidates) {
    try {
      git(`rev-parse --verify --quiet ${upstream}`, { cwd })
    } catch {
      continue
    }
    let range
    try {
      range = git(`rev-list --reverse ${upstream}..HEAD`, { cwd })
    } catch {
      continue
    }
    if (!range) return []
    return range.split('\n').filter(Boolean).map((sha) => ({
      sha,
      message: commitField(cwd, sha, '%s'),
    }))
  }
  return []
}

function commitField(cwd, sha, format) {
  try {
    return git(`log -1 --format=${format} ${sha}`, { cwd })
  } catch {
    return ''
  }
}

/** Size and shape of one commit's diff, in the form evaluateRules expects. */
export function commitDiff(cwd, sha) {
  const files = []
  let linesAdded = 0
  let linesDeleted = 0
  try {
    const numstat = git(`diff-tree --no-commit-id --numstat -r ${sha}`, { cwd })
    for (const line of numstat.split('\n')) {
      const parts = line.split('\t')
      if (parts.length < 3) continue
      const added = parseInt(parts[0], 10)
      const deleted = parseInt(parts[1], 10)
      // Binary files report "-" rather than a count.
      const a = Number.isNaN(added) ? 0 : added
      const d = Number.isNaN(deleted) ? 0 : deleted
      files.push({ path: parts[2], added: a, deleted: d })
      linesAdded += a
      linesDeleted += d
    }
  } catch {
    // A merge commit has no numstat; leave the diff empty.
  }
  return {
    files,
    linesAdded,
    linesDeleted,
    date: commitField(cwd, sha, '%aI'),
    author: commitField(cwd, sha, '%an'),
    email: commitField(cwd, sha, '%ae'),
    message: commitField(cwd, sha, '%s'),
    timestamp: Number(commitField(cwd, sha, '%at')) || 0,
    isMerge: commitParentCount(cwd, sha) > 1,
  }
}

function commitParentCount(cwd, sha) {
  try {
    return git(`rev-list --parents -n 1 ${sha}`, { cwd }).split(' ').length - 1
  } catch {
    return 0
  }
}

/** How many commits this author has in the repository before `sha`. */
export function authorPriorCommits(cwd, email) {
  if (!email) return 0
  try {
    return Number(git(`rev-list --count --author=${email} HEAD`, { cwd })) || 0
  } catch {
    return 0
  }
}
