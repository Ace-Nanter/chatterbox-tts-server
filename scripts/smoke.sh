#!/usr/bin/env bash
# Post-deployment smoke test for tts-server. Usage:
#   ./scripts/smoke.sh [BASE_URL]       (default: $BASE_URL, else http://localhost:4123)
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-http://localhost:4123}}"

fail() { echo "FAIL: $1" >&2; exit 1; }

echo "== GET /health"
health=$(curl -fsS "$BASE_URL/health") || fail "/health unreachable"
echo "$health"
echo "$health" | grep -q '"engine"[[:space:]]*:[[:space:]]*"omnivoice"' \
  || fail "/health does not report engine=omnivoice"
echo "$health" | grep -q '"model_loaded"[[:space:]]*:[[:space:]]*true' \
  || fail "/health reports model_loaded=false"

echo
echo "== GET /languages"
langs=$(curl -fsS "$BASE_URL/languages") || fail "/languages unreachable"
echo "$langs" | grep -q '"French"' || fail "French missing from /languages"
echo "$langs" | grep -q '"English"' || fail "English missing from /languages"
echo "languages: $langs" | head -c 300; echo " …"

echo
echo "== POST /v1/audio/speech (voice=athena, lang_code=fr)"
for fmt in wav mp3 opus; do
  out="/tmp/tts-smoke-$fmt"
  mime=$(curl -fsS -o "$out" -w '%{content_type}' \
    -X POST "$BASE_URL/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{\"input\": \"Bonjour, ceci est un test de synthèse vocale.\", \"voice\": \"athena\", \"lang_code\": \"fr\", \"response_format\": \"$fmt\"}") \
  || fail "synthesis ($fmt) failed"
  [ -s "$out" ] || fail "empty payload for $fmt"
  case "$mime" in
    audio/*) ;;
    *) fail "$fmt: unexpected content-type '$mime'" ;;
  esac
  fmt_mime=$( file --mime-type -b "$out" 2>/dev/null || echo audio/x-wav)
  case "$fmt_mime" in
    audio/mpeg) [[ "$fmt" == "mp3" ]] && echo "$fmt: OK ($mime, file says $fmt_mime, $(wc -c < "$out") bytes)" || fail "$fmt: payload is mp3 but requested $fmt" ;;
    audio/x-flac|audio/flac) [[ "$fmt" == "flac" ]] && echo "$fmt: OK" || fail "$fmt: payload is flac but requested $fmt" ;;
    *) echo "$fmt: OK ($mime, container says $fmt_mime, $(wc -c < "$out") bytes)" ;;
  esac
done

echo
echo "SMOKE OK — all checks passed ($BASE_URL)"
