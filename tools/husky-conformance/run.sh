#!/bin/sh
# Run husky's own test suite against the husky this repository installed.
#
#     sh tools/husky-conformance/run.sh            # skip quietly if node is absent
#     sh tools/husky-conformance/run.sh --require  # ...but fail if node is absent (CI)
#     sh tools/husky-conformance/run.sh --only 9   # one test, for debugging
#
# Why this exists
# ---------------
# Our hooks are invoked by husky, not by us: git runs `.husky/_/<hook>`, which
# sources `.husky/_/h`, which runs `.husky/<hook>`. Every one of our hook tests
# starts *after* that chain, so all of them pass even if the chain itself is
# broken on this machine -- wrong `core.hooksPath`, a `_/` directory generated
# by a different husky version, a shim that is not executable, `HUSKY=0` not
# honoured. Those are exactly the failures that make every hook silently stop
# running.
#
# Upstream's suite is the specification for that chain, so we run it as-is
# rather than restating it. A pass means "the husky we depend on still behaves
# the way our integration assumes, on this OS and this Node".
#
# The upstream files in `upstream/` are unmodified, and are the only third-party
# source in this repository. See README.md for provenance.

set -u

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
UPSTREAM=$HERE/upstream
HUSKY_MODULE=$ROOT/node_modules/husky

# Upstream's own scripts hard-code this path -- `setup()` creates
# /tmp/husky-test-<name> and `install()` runs `npm install ../husky.tgz` -- so
# the tarball has to be exactly here. Under Git Bash, /tmp is the Windows temp
# directory, which is what makes the native `npm` in the tests able to resolve
# it.
TMPDIR_HUSKY=/tmp
TARBALL=$TMPDIR_HUSKY/husky.tgz
LOGDIR=$TMPDIR_HUSKY/husky-conformance-logs

TESTS='1_default 2_in-sub-dir 3_from-sub-dir 4_not-git-dir
5_git_command_not_found 6_command_not_found 7_node_modules_path 8_set_u
9_husky_0 10_init 11_time 12_deprecated'

# Split from the list above, which stays in upstream's run order.
TEST_NAMES=$TESTS

# What each test pins down. Printed in the report so the output is a map of the
# contract rather than a wall of dots.
describe() {
	case $1 in
	1_default) echo 'install sets core.hooksPath=.husky/_ and a failing hook blocks the commit' ;;
	2_in-sub-dir) echo 'a custom hooks directory (husky sub/husky) is honoured' ;;
	3_from-sub-dir) echo 'install from a subdirectory via the prepare script works' ;;
	4_not-git-dir) echo 'install neither throws nor half-writes outside a repository' ;;
	5_git_command_not_found) echo 'the Node API degrades to a message when git is off PATH' ;;
	6_command_not_found) echo 'a hook exiting 127 is reported, not swallowed' ;;
	7_node_modules_path) echo 'node_modules/.bin is on PATH inside a hook' ;;
	8_set_u) echo 'a `set -u` init.sh does not break hook dispatch' ;;
	9_husky_0) echo 'HUSKY=0 skips install, and skips hooks when set by init.sh' ;;
	10_init) echo '`husky init` completes' ;;
	11_time) echo 'a hook adds no perceptible latency to a commit' ;;
	12_deprecated) echo 'a v8-era shim header warns instead of failing' ;;
	*) echo 'unknown test' ;;
	esac
}

# Tests we knowingly do not require to pass on every platform.
#
# Empty on purpose: every test is expected to pass everywhere. Anything added
# here needs a reason in `allow_reason`, and the report shouts when an allowed
# test starts passing, so the list cannot quietly rot into a bin of real
# failures.
allow_reason() {
	case $1 in
	*) : ;;
	esac
}

# -- arguments ---------------------------------------------------------

REQUIRE=0
ONLY=
while [ $# -gt 0 ]; do
	case $1 in
	--require) REQUIRE=1 ;;
	--only)
		shift
		[ $# -gt 0 ] || {
			printf '%s\n' "husky-conformance: --only needs a test name" >&2
			exit 2
		}
		ONLY=$1
		;;
	--only=*) ONLY=${1#--only=} ;;
	-h | --help)
		sed -n '2,18p' "$0"
		exit 0
		;;
	*)
		printf '%s\n' "husky-conformance: unknown argument: $1" >&2
		exit 2
		;;
	esac
	shift
done

# -- prerequisites -----------------------------------------------------

missing=
command -v node >/dev/null 2>&1 || missing="node"
command -v npm >/dev/null 2>&1 || { [ -n "$missing" ] && missing="$missing, "; missing="${missing}npm"; }
command -v git >/dev/null 2>&1 || { [ -n "$missing" ] && missing="$missing, "; missing="${missing}git"; }
[ -d "$HUSKY_MODULE" ] || { [ -n "$missing" ] && missing="$missing, "; missing="${missing}node_modules/husky"; }

if [ -n "$missing" ]; then
	printf '%s\n' "husky-conformance: skipped, not installed: $missing"
	if [ "$REQUIRE" = 1 ]; then
		printf '%s\n' "husky-conformance: --require was given, so this is a failure"
		exit 1
	fi
	exit 0
fi

# -- pack the module under test ---------------------------------------
#
# `npm pack` here would pack *this* repository. The suite wants a husky
# tarball, so we pack the installed module instead, which is the same bytes a
# consumer would get from the registry.
if ! (cd "$ROOT" && npm pack "$HUSKY_MODULE" --pack-destination "$TMPDIR_HUSKY" >/dev/null 2>&1); then
	printf '%s\n' "husky-conformance: npm pack failed for $HUSKY_MODULE" >&2
	exit 1
fi
# The filename carries the version, which is the point of pinning nothing here:
# whatever version is installed is the version under test.
packed=$(ls "$TMPDIR_HUSKY"/husky-*.tgz 2>/dev/null | head -n 1)
if [ -z "$packed" ]; then
	printf '%s\n' "husky-conformance: npm pack produced no husky-*.tgz" >&2
	exit 1
fi
mv -f "$packed" "$TARBALL"
husky_version=$(basename "$packed" .tgz)

rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

printf '\n%s\n' "husky conformance -- $husky_version -- $(uname -s 2>/dev/null || echo unknown) -- node $(node --version)"
printf '%-26s %-6s %s\n' TEST RESULT 'WHAT IT PINS DOWN'
printf '%s\n' '-------------------------------------------------------------------------------'

failures=0
allowed=0
ran=0
unexpected_pass=
failed_names=

for name in $TEST_NAMES; do
	if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then
		continue
	fi
	ran=$((ran + 1))
	log=$LOGDIR/$name.log
	if (cd "$UPSTREAM" && sh "test/$name.sh") >"$log" 2>&1; then
		status=pass
	else
		status=FAIL
	fi

	if [ "$status" = pass ]; then
		if [ -n "$(allow_reason "$name")" ]; then
			printf '%-26s %-6s %s\n' "$name" "PASS*" "$(describe "$name")"
			unexpected_pass="$unexpected_pass $name"
		else
			printf '%-26s %-6s %s\n' "$name" pass "$(describe "$name")"
		fi
	else
		reason=$(allow_reason "$name")
		if [ -n "$reason" ]; then
			printf '%-26s %-6s %s\n' "$name" "fail*" "$reason"
			allowed=$((allowed + 1))
		else
			printf '%-26s %-6s %s\n' "$name" FAIL "$(describe "$name")"
			failures=$((failures + 1))
			failed_names="$failed_names $name"
		fi
	fi
done

printf '%s\n' '-------------------------------------------------------------------------------'

if [ $failures -gt 0 ]; then
	printf '\n%s\n' "$failures test(s) failed. Tail of each log:"
	for name in $failed_names; do
		printf '\n--- %s ---\n' "$name"
		tail -n 20 "$LOGDIR/$name.log"
	done
	printf '\nFull logs: %s\n' "$LOGDIR"
fi

if [ -n "$unexpected_pass" ]; then
	printf '\n%s\n' "These are listed as allowed to fail but passed:$unexpected_pass"
	printf '%s\n' "Remove them from allow_reason() so the list stays honest."
fi

if [ $failures -gt 0 ]; then
	exit 1
fi
printf '\n%s\n' "$ran conformance test(s) ran, $allowed allowed to fail, 0 unexpected"
exit 0
