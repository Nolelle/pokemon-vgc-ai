#!/usr/bin/env bash
# Check out public smogon/pokemon-showdown at the commit the mechanics catalog is pinned
# to (data/champions/mechanics_catalog.json -> generated_from.showdown_commit).
#
#   checkout_pinned_showdown.sh <dest> [--build] [--history]
#
# --build    npm ci + `node build --force` (needed to run battles / validate-team).
# --history  keep full commit history (blobless) so the parity check can list the
#            upstream commits the pin is missing; otherwise a depth-1 fetch is used.
set -euo pipefail

dest="$1"
shift
build=false
history=false
for arg in "$@"; do
  case "$arg" in
    --build) build=true ;;
    --history) history=true ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
pin="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["generated_from"]["showdown_commit"])' \
  "$repo_root/data/champions/mechanics_catalog.json")"
echo "pinned Showdown commit: $pin"

if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD)" = "$pin" ]; then
  echo "already at pin (cache hit)"
else
  rm -rf "$dest"
  if [ "$history" = true ]; then
    git clone --quiet --filter=blob:none https://github.com/smogon/pokemon-showdown.git "$dest"
    git -C "$dest" checkout --quiet "$pin"
  else
    git init --quiet "$dest"
    git -C "$dest" remote add origin https://github.com/smogon/pokemon-showdown.git
    git -C "$dest" fetch --quiet --depth 1 origin "$pin"
    git -C "$dest" checkout --quiet FETCH_HEAD
  fi
fi

if [ "$build" = true ] && [ ! -f "$dest/dist/sim/index.js" ]; then
  (cd "$dest" && npm ci --no-audit --no-fund && node build --force)
fi
