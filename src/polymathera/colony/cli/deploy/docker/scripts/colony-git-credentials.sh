#!/bin/sh
# Per-agent git credential helper. Prints credentials to stdout in
# git's credential-helper protocol when invoked by git for an HTTPS
# operation against github.com.
#
# The token in $COLONY_GIT_CREDENTIALS_FILE is minted at agent
# startup (and periodically refreshed) by the agent process — see
# colony/distributed/git_credentials.py. The shell helper is
# intentionally dumb: it just reads the file. When the file is
# missing or empty (no per-tenant App installation configured),
# the helper prints nothing and git surfaces its own
# "Authentication failed" error, which the auth-error classifier
# in colony/distributed/stores/git.py reshapes into a typed
# GitAuthError.
#
# Wired into git's system config via Dockerfile.base.
set -eu

token_file="${COLONY_GIT_CREDENTIALS_FILE:-/tmp/colony-git-credentials}"

if [ -r "$token_file" ] && [ -s "$token_file" ]; then
    echo "username=x-access-token"
    printf 'password='
    cat "$token_file"
    echo
fi
