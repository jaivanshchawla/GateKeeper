// Husky installer used by the `prepare` script.
//
// Adopted from husky's documented guidance. `"prepare": "husky"` fails in an
// environment that installs only production dependencies, because husky is a
// devDependency and is therefore absent -- the install then dies with
// "command not found". CI servers and Docker images almost never need git
// hooks either, so the install is skipped there on purpose.
//
// HUSKY=1 forces the install even in CI, which is how the hooks workflow
// verifies the real setup path instead of trusting it.
const husky = process.env.HUSKY
const forced = husky === '1'
const skipped =
  husky === '0' ||
  (!forced && (process.env.NODE_ENV === 'production' || process.env.CI === 'true'))

if (skipped) {
  console.log('husky: install skipped (HUSKY=0, NODE_ENV=production or CI)')
  process.exit(0)
}

try {
  // Imported dynamically so a production install without husky still succeeds.
  const install = (await import('husky')).default
  // install() returns '' on success, or a message explaining why it did not.
  const result = install()
  console.log(result ? `husky: ${result}` : 'husky: hooks installed (core.hooksPath -> .husky/_)')
} catch (error) {
  console.log(`husky: not installed (${error.message}); git hooks not set up`)
}
