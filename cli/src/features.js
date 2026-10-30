/**
 * The feature contract the model was trained on, computed from a local diff.
 *
 * `ml/history_features.py::compute_change_shape` is the reference for the
 * change-shape fields, and these formulas are copied from it on purpose: a
 * second, subtly different definition would make the CLI's scores
 * incomparable with the API's for no visible reason. Features that need
 * repository history the CLI does not have (per-file priors, author-file
 * familiarity) are omitted rather than guessed -- the API defaults them to 0,
 * so a CLI score is a first-class model score, just from a smaller vector.
 */

const TEST_MARKERS = ['test', 'spec', '_test.', '_spec.', 'tests/', 'test_', '__tests__']
const CONFIG_MARKERS = [
  '.yaml', '.yml', '.toml', '.lock', 'dockerfile', '.github/',
  'docker-compose', 'makefile', '.env', '.ini', '.cfg',
]
// Same set as ml/extract_features.py: a message mentioning any of these is a
// fix/revert signal, which the model was trained to weight.
const FIX_WORDS = ['fix', 'bug', 'revert']

function directoryOf(path) {
  const index = path.lastIndexOf('/')
  return index === -1 ? '.' : path.slice(0, index)
}

export function buildFeatures(diff, { authorPriorCommits = 0 } = {}) {
  const files = diff.files || []
  const linesAdded = diff.linesAdded || 0
  const linesDeleted = diff.linesDeleted || 0
  const filesTouched = files.length
  const dirs = new Set(files.map((f) => directoryOf(f.path)))
  const dirsTouched = dirs.size

  let testCount = 0
  let configCount = 0
  for (const file of files) {
    const lower = file.path.toLowerCase()
    if (TEST_MARKERS.some((marker) => lower.includes(marker))) testCount += 1
    if (CONFIG_MARKERS.some((marker) => lower.includes(marker))) configCount += 1
  }

  const message = diff.message || ''
  const when = diff.timestamp ? new Date(diff.timestamp * 1000) : new Date(diff.date || Date.now())

  return {
    lines_added: linesAdded,
    lines_deleted: linesDeleted,
    files_touched: filesTouched,
    dirs_touched: dirsTouched,
    author_prior_commits: authorPriorCommits,
    hour_of_day: when.getUTCHours(),
    day_of_week: (when.getUTCDay() + 6) % 7, // Monday=0, matching the training data
    commit_msg_length: message.length,
    is_fix_bug_revert: FIX_WORDS.some((word) => message.toLowerCase().includes(word)) ? 1 : 0,

    // Change shape (ml/history_features.py::compute_change_shape).
    churn_ratio: linesDeleted / (linesAdded + 1),
    change_entropy: filesTouched > 1 ? Math.log2(filesTouched) : 0,
    max_file_churn: filesTouched > 0 ? (linesAdded + linesDeleted) / filesTouched : 0,
    is_test_only: filesTouched > 0 && testCount === filesTouched ? 1 : 0,
    test_to_code_ratio: filesTouched > 0 ? testCount / filesTouched : 0,
    config_touch: configCount > 0 ? 1 : 0,
    is_merge: diff.isMerge ? 1 : 0,
    files_per_dir_ratio: dirsTouched > 0 ? filesTouched / dirsTouched : 0,
  }
}
