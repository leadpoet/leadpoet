#!/bin/bash
# Run after checkout, before processing artifacts or obtaining AWS credentials.
set -Eeuo pipefail

test "${GITHUB_REPOSITORY:-}" = leadpoet/leadpoet
test "${GITHUB_EVENT_NAME:-}" = workflow_dispatch
test "${GITHUB_REF:-}" = refs/heads/main
[[ "${GITHUB_SHA:-}" =~ ^[0-9a-f]{40}$ ]]
test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
git fetch --no-tags origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
