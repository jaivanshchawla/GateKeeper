# Shared bootstrap for the Gatekeeper husky hooks.
#
# Sourced by each .husky/<hook> shim:
#     . "$(dirname "$0")/lib/bootstrap.sh"
#     gatekeeper_run_hook pre-commit
#
# Two jobs, both about surviving environments husky does not control:
#
# 1. Find a Python that actually works. `command -v python3` succeeds on
#    Windows even when the only match is the Microsoft Store "App execution
#    alias" stub, which exits non-zero and prints an install advert, so each
#    candidate is *probed* rather than trusted.
# 2. Stay quiet and harmless when it cannot. See the note on sourcing below.
#
# Which interpreter runs the project's own tooling (venv-aware) is a separate
# question, answered in one place: scripts/hooks/_runtime.py.

# husky's dispatcher sets -x on HUSKY=2, but the hook it hands over to runs in
# a *new* shell, so the trace stops at exactly the point where the interesting
# work starts. Continuing it here, before anything else, keeps `HUSKY=2` a
# single switch over the whole chain: dispatcher, shim, bootstrap, Python.
[ "${HUSKY-}" = "2" ] && set -x

# Locate the .husky/ directory.
#
# git invokes a hook by path, so `$0` is normally the hook file and its
# directory is what we want. When this file is sourced some other way -- for
# example `sh -c '. .husky/lib/bootstrap.sh'` -- fall back to the hooks
# directory in the repository root.
#
# The `[ -f ]` test is load-bearing. POSIX requires a non-interactive shell to
# abort when `.` cannot find its file, so `command || true` cannot protect it:
# the shell is already gone by the time `||` would run. An earlier version
# sourced unconditionally and a wrong `$0` silently took every hook down with
# no error at all.
gatekeeper_lib_dir() {
	dir=$(dirname "$0" 2>/dev/null)
	if [ -n "$dir" ] && [ -f "$dir/lib/common.sh" ]; then
		printf '%s\n' "$dir"
		return 0
	fi
	root=$(git rev-parse --show-toplevel 2>/dev/null) || return 1
	[ -f "$root/.husky/lib/common.sh" ] || return 1
	printf '%s\n' "$root/.husky"
}

_gatekeeper_lib=$(gatekeeper_lib_dir) || _gatekeeper_lib=
if [ -n "$_gatekeeper_lib" ]; then
	. "$_gatekeeper_lib/lib/common.sh"
fi

# True when an interpreter can actually start and is Python 3.8+.
gatekeeper_python_works() {
	command -v "$1" >/dev/null 2>&1 &&
		"$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1
}

gatekeeper_run_hook() {
	name=$1
	shift

	root=$(git rev-parse --show-toplevel 2>/dev/null) || root=.
	cd "$root" || exit 1

	# Hook names use dashes (pre-commit); the modules use underscores
	# (scripts.hooks.pre_commit).
	module=$(printf '%s' "$name" | tr '-' '_')

	# GATEKEEPER_PYTHON is tried first so the escape hatch actually works when
	# nothing usable is on PATH. It was documented long before it was honoured.
	for candidate in "${GATEKEEPER_PYTHON:-}" python python3 py; do
		[ -n "$candidate" ] || continue
		if gatekeeper_python_works "$candidate"; then
			# -m (not a file path) so the hook modules are imported as part of
			# the scripts.hooks package and need no sys.path bootstrapping of
			# their own.
			exec "$candidate" -m "scripts.hooks.$module" "$@"
		fi
	done

	echo "gatekeeper-hooks: no working Python 3 found on PATH." >&2
	echo "gatekeeper-hooks: set GATEKEEPER_PYTHON, or install Python 3.8+." >&2
	exit 1
}
