#!/usr/bin/env node
/**
 * Gatekeeper CLI — commit risk scoring from the command line.
 *
 * The bin is deliberately thin: parse argv, call a command, exit with its
 * code. Everything testable lives in the sibling modules.
 */

import {
  cmdAuth,
  cmdCheck,
  cmdConfig,
  cmdExplain,
  cmdInit,
  cmdStatus,
  cmdWhoami,
  version,
} from './commands.js'
import { resolveApiUrl } from './api.js'

const USAGE = `
🛡️  Gatekeeper CLI — ML commit-risk scoring for every push

Usage:
  gatekeeper init                Install the pre-push hook and .gatekeeper.yml
  gatekeeper check [--json]      Score unpushed commits (rules + model)
  gatekeeper status              Open findings for this repository
  gatekeeper explain <sha>       Why one commit scored the way it did
  gatekeeper config [--api-url <url>]  Show or set configuration
  gatekeeper auth --token <jwt>  Save a Clerk session token (or --clear)
  gatekeeper whoami              Show who the API thinks you are
  gatekeeper --version           Print the CLI version

Environment:
  GATEKEEPER_API_URL   API base URL (or: gatekeeper config --api-url ...)
  GATEKEEPER_TOKEN     Session token for a Clerk-protected API
  SKIP=gatekeeper-score  Skip the pre-push hook once

Examples:
  npx gatekeeper-cli init
  npx gatekeeper-cli check --json
  gatekeeper config --api-url https://gatekeeper.example.com
`

async function main(argv) {
  const [command, ...rest] = argv

  switch (command) {
    case 'init':
      return cmdInit({})
    case 'check':
      return cmdCheck({ argv: rest })
    case 'status':
      return cmdStatus({})
    case 'explain':
      return cmdExplain(rest[0], {})
    case 'config':
      return cmdConfig(rest, {})
    case 'auth':
      return cmdAuth(rest, {})
    case 'whoami':
      return cmdWhoami({})
    case '--version':
    case '-v':
    case 'version':
      console.log(version())
      return 0
    case '--help':
    case '-h':
    case 'help':
      console.log(USAGE)
      return 0
    case undefined:
      console.log(USAGE)
      if (!resolveApiUrl()) {
        console.log('ℹ  No API URL configured yet — `gatekeeper config --api-url <url>` for model scoring.')
      }
      return 0
    default:
      console.error(`Unknown command: ${command}`)
      console.log(USAGE)
      return 1
  }
}

const code = await main(process.argv.slice(2))
process.exitCode = code
