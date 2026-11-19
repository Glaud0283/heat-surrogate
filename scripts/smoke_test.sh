#!/usr/bin/env bash
# Usage: scripts/smoke_test.sh BASE_URL [EXPECTED_GIT_SHA]
# Passes only if the service answers, runs the expected commit, and predicts sensible physics.
set -euo pipefail

url="${1%/}"
expected="${2:-}"
case "$url" in http*) ;; *) url="https://$url" ;; esac

health=""
for i in $(seq 1 30); do
  if health=$(curl -fsS --max-time 5 "$url/health"); then break; fi
  echo "waiting for $url/health ($i/30)"
  sleep 5
done
if [ -z "$health" ]; then
  echo "::error::$url/health never answered"
  exit 1
fi
echo "health: $health"

running=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["git_sha"])' <<< "$health")
if [ -n "$expected" ] && [ "$running" != "$expected" ]; then
  echo "::error::service runs $running, expected $expected"
  exit 1
fi

curl -fsS --max-time 10 -X POST "$url/predict" -H "Content-Type: application/json" \
  -d '{"k0": 5, "g": 0.8, "q": 300, "t_top": 20, "t_bottom": 80, "t_left": 50, "t_right": 50}' \
  | python3 -c '
import json, sys
f = json.load(sys.stdin)["field"]
top, bottom = sum(f[0]) / 64, sum(f[-1]) / 64
print(f"top row mean {top:.1f}, bottom row mean {bottom:.1f}")
assert len(f) == 64 and all(len(r) == 64 for r in f), "field is not 64 x 64"
assert top < bottom, "top edge is 20 and bottom edge is 80, so the top row must be cooler"
'
echo "smoke test passed"
