#!/usr/bin/env bash
# Re-render paper HTML/DOCX from the Markdown source, copy the HTML into
# docs/index.html (the GitHub Pages root), and stage everything for commit.
# GitHub Pages picks up the new content on the next push to main.
set -euo pipefail
cd "$(dirname "$0")/.."

( cd paper && pandoc skeleton.md \
    --citeproc \
    --bibliography=references.bib \
    --standalone --number-sections \
    --metadata title="Mining Frequent Failure Sequences in Operational Event Logs" \
    --metadata date="$(date +%Y-%m-%d)" \
    --css=style.css --embed-resources \
    -o skeleton.html )

pandoc paper/skeleton.md \
  --citeproc \
  --bibliography=paper/references.bib \
  --standalone --number-sections \
  --metadata title="Mining Frequent Failure Sequences in Operational Event Logs" \
  --metadata date="$(date +%Y-%m-%d)" \
  -o paper/skeleton.docx

cp paper/skeleton.html docs/index.html
touch docs/.nojekyll

echo "Rebuilt paper/skeleton.{html,docx} and refreshed docs/index.html."
echo "Live at https://apartsinprojects.github.io/FailurePatterns/ after next push."
