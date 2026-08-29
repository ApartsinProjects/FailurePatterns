#!/usr/bin/env bash
# Re-render paper HTML/DOCX from the Markdown source, generate a PDF, copy the
# HTML into docs/index.html (the GitHub Pages root) with a top-right PDF/DOCX
# download bar, and stage the downloadable artifacts alongside it.
# GitHub Pages picks up the new content on the next push to main.
set -euo pipefail
cd "$(dirname "$0")/.."

TODAY="$(date +%Y-%m-%d)"
# Title, author, and keywords come from the YAML front matter in skeleton.md.
# Section numbers are written manually in the headings (so the in-text
# section cross-references stay stable), so pandoc's --number-sections is
# intentionally OFF to avoid double numbering.

# --- Standalone HTML (embedded resources) ---
( cd paper && pandoc skeleton.md \
    --citeproc \
    --bibliography=references.bib \
    --standalone \
    --metadata date="$TODAY" \
    --css=style.css --embed-resources \
    -o skeleton.html )

# --- Word (.docx) ---
pandoc paper/skeleton.md \
  --citeproc \
  --bibliography=paper/references.bib \
  --standalone \
  --metadata date="$TODAY" \
  -o paper/skeleton.docx

# --- PDF via headless Chrome/Edge (rendered from the clean standalone HTML,
#     before the download bar is injected, so the PDF carries no web chrome) ---
CHROME=""
for c in \
  "/c/Program Files/Google/Chrome/Application/chrome.exe" \
  "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
  "/c/Program Files/Microsoft/Edge/Application/msedge.exe" \
  "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"; do
  [ -f "$c" ] && CHROME="$c" && break
done

# A FRESH, isolated user-data-dir each run avoids attaching to an already-running
# browser and avoids a stale profile lock (either makes Chrome return before the
# page renders, producing a near-empty PDF). --virtual-time-budget waits for the
# embedded resources to load before printing. Chrome is a native Windows binary,
# so every path handed to it must be Windows form (cygpath -m: E:/... , mixed
# slashes); a POSIX /e/... or /tmp/... path silently loads nothing.
if [ -n "$CHROME" ]; then
  HTML_WIN="$(cygpath -m "$(pwd)/paper/skeleton.html")"
  HTML_URL="file:///$(printf '%s' "$HTML_WIN" | sed 's/ /%20/g')"
  PDF_WIN="$(cygpath -m "$(pwd)/docs/paper.pdf")"
  PDF_PROFILE_POSIX="$(mktemp -d "${TMPDIR:-/tmp}/ffp-chrome-XXXXXX")"
  PDF_PROFILE_WIN="$(cygpath -m "$PDF_PROFILE_POSIX")"
  "$CHROME" --headless=new --disable-gpu --no-first-run --no-pdf-header-footer \
    --user-data-dir="$PDF_PROFILE_WIN" --virtual-time-budget=20000 \
    --print-to-pdf="$PDF_WIN" "$HTML_URL" >/dev/null 2>&1 \
    && echo "Generated docs/paper.pdf via headless browser." \
    || echo "WARNING: headless PDF generation failed; keeping previous docs/paper.pdf." >&2
  rm -rf "$PDF_PROFILE_POSIX"
else
  echo "WARNING: no Chrome/Edge found; skipping PDF regeneration." >&2
fi

# --- Stage downloadable artifacts + the Pages root ---
cp paper/skeleton.docx docs/paper.docx
cp paper/skeleton.html docs/index.html
touch docs/.nojekyll

# --- Inject the top-right PDF/DOCX download bar into the Pages HTML ---
PY=""
for c in /c/Python314/python python py; do command -v "$c" >/dev/null 2>&1 && PY="$c" && break; done
"$PY" scripts/inject_download_bar.py docs/index.html

echo "Rebuilt paper/skeleton.{html,docx}, docs/paper.{pdf,docx}, and docs/index.html (with download bar)."
echo "Live at https://apartsinprojects.github.io/FailurePatterns/ after next push."
