# Shared shell helpers, sourced by bootstrap.sh for every hook.
#
# The tty guard is husky's documented workaround for Windows: under Git Bash
# with some GUI clients a hook's stdin is not a usable tty, and interactive
# tools then fail with "stdin is not a tty".
#
# Gatekeeper hooks are non-interactive, so the guard is opt-in rather than
# automatic, and it must NEVER be used by a hook that reads stdin --
# replacing stdin with /dev/tty would silently discard git's payload.
# `pre-push` (ref lines) and `post-rewrite` (sha pairs) both read stdin and
# therefore must not call this.

command_exists() {
	command -v "$1" >/dev/null 2>&1
}

gatekeeper_restore_tty() {
	if command_exists winpty && test -t 1; then
		exec < /dev/tty
	fi
}
