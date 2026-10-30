import assert from 'node:assert/strict'
import { test } from 'node:test'
import { repoNameFromRemote } from '../src/git.js'

test('https, ssh and trailing-slash remotes all name the same repository', () => {
  for (const url of [
    'https://github.com/jaivanshchawla/GateKeeper.git',
    'https://github.com/jaivanshchawla/GateKeeper',
    'git@github.com:jaivanshchawla/GateKeeper.git',
    'https://github.com/jaivanshchawla/GateKeeper/',
  ]) {
    assert.equal(repoNameFromRemote(url), 'jaivanshchawla/GateKeeper', url)
  }
})

test('a non-GitHub remote still yields a repository name', () => {
  assert.equal(repoNameFromRemote('https://gitlab.com/team/project.git'), 'project')
  assert.equal(repoNameFromRemote('git@gitlab.com:team/project.git'), 'project')
})

test('no remote is null, not the string "null"', () => {
  assert.equal(repoNameFromRemote(null), null)
  assert.equal(repoNameFromRemote(''), null)
})
