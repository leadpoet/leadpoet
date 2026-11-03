#!/bin/bash
# Inputs arrive through environment values, never through generated shell text.
set -Eeuo pipefail

build_commit="${GITHUB_SHA:?}"
mode=publish
if [ -n "${REPRODUCE_COMMIT:-}" ]; then
  [[ "$REPRODUCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
    echo 'ERROR: reproduce_commit must be a full lowercase Git commit' >&2
    exit 1
  }
  test "$(git rev-parse "${REPRODUCE_COMMIT}^{commit}")" = "$REPRODUCE_COMMIT"
  git merge-base --is-ancestor "$REPRODUCE_COMMIT" origin/main
  build_commit="$REPRODUCE_COMMIT"
  mode=reproduce
fi
if [ -n "${EXPECTED_PCR0:-}" ]; then
  test "$mode" = reproduce
  [[ "$EXPECTED_PCR0" =~ ^[0-9a-f]{96}$ ]] || exit 1
  test "$EXPECTED_PCR0" != "$(printf '%096d' 0)"
fi
printf 'build_commit=%s\nmode=%s\n' "$build_commit" "$mode" >> "${GITHUB_OUTPUT:?}"
