# Shared bootstrap for the Gatekeeper husky hooks.
#
# Sourced by each .husky/<hook> shim:
#     . "$(dirname "$0")/lib/bootstrap.sh"
#     gatekeeper_run_hook pre-commit
#
# Why this exists: `command -v python3` succeeds on Windows even when the
# only match is the Microsoft Store "App execution alias" stub, which exits
# non-zero and prints an install prompt. Husky gives us sh but no promise of
# Python, so each candidate is *probed* rather than trusted.
#
# This only decides which interpreter can start the hook script. Which
# interpreter runs the project's own tooling (venv-aware) is decided in one
# place: scripts/hooks/_runtime.py.

gatekeeper_run_hook() {
	name=$1
	shift

	root=$(git rev-parse --show-toplevel 2>/dev/null) || root=.
	cd "$root" || exit 1

	# Hook names use dashes (pre-commit); the modules use underscores
	# (scripts.hooks.pre_commit).
	module=$(printf '%s' "$name" | tr '-' '_')

	[ "${HUSKY-}" = "2" ] && set -x

	for candidate in python python3 py; do
		if command -v "$candidate" >/dev/null 2>&1 &&
			"$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
			# -m (not a file path) so the hook modules are imported as part
			# of the scripts.hooks package and need no sys.path bootstrapping
			# of their own.
			exec "$candidate" -m "scripts.hooks.$module" "$@"
		fi
	done

	echo "gatekeeper-hooks: no working Python 3 found on PATH." >&2
	echo "gatekeeper-hooks: set GATEKEEPER_PYTHON, or install Python 3.8+." >&2
	exit 1
}
