#!/usr/bin/env bash
# Publish the shareable charts to the `charts` branch of this repository by hand.
#
# The scheduled workflow does this after every daily run. Use this script to
# publish straight away, for example after chart names change, so that the
# README images load. It builds the branch in a throwaway folder and replaces
# the branch's contents, so nothing in your working copy is touched.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${CHARTS_REPO_URL:-}" ]; then
  repo_url="$CHARTS_REPO_URL"
else
  repo_url="$(git remote get-url origin)"
  case "$repo_url" in
    *football-corners-predictor*) ;;
    *) echo "origin is $repo_url, not football-corners-predictor. Run this from the code repository." >&2; exit 1 ;;
  esac
fi

if [ ! -d results ]; then
  echo "No results/ folder yet. Run 'python -m corners report' first." >&2
  exit 1
fi

out="$(mktemp -d)"
python -m corners charts "$out"   # exits non-zero if there are no charts to publish

cd "$out"
git init -q -b charts
git add -A
git commit -q -m "Charts $(date -u +%F)"
git push --force "$repo_url" charts
echo "Published the charts branch to $repo_url"
