#!/usr/bin/env bash
# Fetch the SPMF v2.64 jar (Java-21-compatible) into scripts/spmf.jar.
# See docs/tools.md for why 2.64 and not 2.66.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://www.philippe-fournier-viger.com/spmf/spmf2.64.jar"
EXPECTED_SHA256="c67c2e56bdd2072eadc0e4bc96c8221cec38c9109f99c4b5f10be8eba64ec250"
curl -sSL -o spmf.jar "$URL"
GOT=$(sha256sum spmf.jar | awk '{print $1}')
if [ "$GOT" != "$EXPECTED_SHA256" ]; then
    echo "SHA256 mismatch: expected $EXPECTED_SHA256, got $GOT" >&2
    exit 1
fi
echo "spmf.jar fetched, SHA256 verified."
