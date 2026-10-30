import assert from 'node:assert/strict'
import { test } from 'node:test'
import { buildFeatures } from '../src/features.js'

const when = Math.floor(Date.parse('2026-03-04T23:30:00Z') / 1000) // Wednesday, late

const diff = (overrides = {}) => ({
  files: [
    { path: 'src/app.py', added: 30, deleted: 10 },
    { path: 'src/lib/util.py', added: 10, deleted: 0 },
  ],
  linesAdded: 40,
  linesDeleted: 10,
  message: 'Add retry to the poller',
  timestamp: when,
  ...overrides,
})

test('the required model fields are all present and non-negative', () => {
  const features = buildFeatures(diff(), { authorPriorCommits: 7 })
  for (const key of [
    'lines_added', 'lines_deleted', 'files_touched', 'dirs_touched',
    'author_prior_commits', 'hour_of_day', 'day_of_week', 'commit_msg_length',
    'is_fix_bug_revert',
  ]) {
    assert.equal(typeof features[key], 'number', key)
    assert.ok(features[key] >= 0, key)
  }
  assert.equal(features.author_prior_commits, 7)
  assert.equal(features.files_touched, 2)
  assert.equal(features.dirs_touched, 2)
})

test('day_of_week is Monday=0 like the training data', () => {
  const wednesday = buildFeatures(diff(), {})
  assert.equal(wednesday.day_of_week, 2)
  const sunday = buildFeatures(
    diff({ timestamp: Math.floor(Date.parse('2026-03-08T12:00:00Z') / 1000) }),
    {},
  )
  assert.equal(sunday.day_of_week, 6)
})

test('utc hour is used so a laptop timezone cannot change the score', () => {
  assert.equal(buildFeatures(diff(), {}).hour_of_day, 23)
})

test('change shape matches ml/history_features.py', () => {
  const features = buildFeatures(diff(), {})
  // churn_ratio = deleted / (added + 1)
  assert.equal(features.churn_ratio, 10 / 41)
  // change_entropy = log2(files)
  assert.equal(features.change_entropy, 1)
  // max_file_churn = (added + deleted) / files
  assert.equal(features.max_file_churn, 25)
  assert.equal(features.files_per_dir_ratio, 1)
})

test('a single-file change has zero entropy', () => {
  const features = buildFeatures(
    diff({ files: [{ path: 'a.py', added: 1, deleted: 1 }], linesAdded: 1, linesDeleted: 1 }),
    {},
  )
  assert.equal(features.change_entropy, 0)
})

test('test-only and config-touching changes are flagged', () => {
  const testsOnly = buildFeatures(
    diff({
      files: [{ path: 'tests/test_app.py', added: 5, deleted: 0 }],
      linesAdded: 5,
      linesDeleted: 0,
    }),
    {},
  )
  assert.equal(testsOnly.is_test_only, 1)
  assert.equal(testsOnly.test_to_code_ratio, 1)

  const config = buildFeatures(
    diff({ files: [{ path: '.github/workflows/ci.yml', added: 1, deleted: 0 }], linesAdded: 1, linesDeleted: 0 }),
    {},
  )
  assert.equal(config.config_touch, 1)
  assert.equal(config.is_test_only, 0)
})

test('fix/bug/revert in the message sets is_fix_bug_revert', () => {
  assert.equal(buildFeatures(diff({ message: 'Fix crash on empty input' }), {}).is_fix_bug_revert, 1)
  assert.equal(buildFeatures(diff({ message: 'Bump version' }), {}).is_fix_bug_revert, 0)
})

test('a merge is marked as one, and merge commits do not produce features from nothing', () => {
  const merge = buildFeatures(
    diff({ files: [], linesAdded: 0, linesDeleted: 0, isMerge: true }),
    {},
  )
  assert.equal(merge.is_merge, 1)
  assert.equal(merge.files_touched, 0)
  assert.equal(merge.churn_ratio, 0)
  assert.equal(merge.files_per_dir_ratio, 0)
})
