// Reverse of the husky install.
//
// husky v9 removed `husky uninstall` and only documents the manual step
// ("git config --unset core.hooksPath"), which is easy to run incorrectly:
// unsetting the key restores .git/hooks/ but leaves the generated .husky/_
// directory behind, so the next install starts from stale shims.
import { execFileSync } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

// Resolved from this file rather than the cwd, so the command behaves the
// same when npm runs it from the repo root and when it is run directly.
const huskyDir = dirname(fileURLToPath(import.meta.url))

let unset = false
try {
  execFileSync('git', ['config', '--unset', 'core.hooksPath'], { stdio: 'ignore' })
  unset = true
} catch {
  // Already unset, or not a git repository.
}

const generated = join(huskyDir, '_')
const removed = existsSync(generated)
if (removed) rmSync(generated, { recursive: true, force: true })

console.log(
  unset
    ? 'husky: core.hooksPath unset — .git/hooks/ is active again'
    : 'husky: core.hooksPath was not set'
)
if (removed) console.log('husky: removed generated .husky/_/')
