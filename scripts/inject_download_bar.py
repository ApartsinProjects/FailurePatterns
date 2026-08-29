"""Inject a fixed top-right download bar (PDF + DOCX) into docs/index.html.

Idempotent: removes any prior injected bar before adding the current one,
so re-running publish_paper.sh does not stack duplicates. The bar is
hidden in print/PDF output via a scoped @media print rule.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

MARKER_START = "<!-- download-bar:start -->"
MARKER_END = "<!-- download-bar:end -->"

BAR = f"""{MARKER_START}
<style>
.dl-bar{{position:fixed;top:12px;right:14px;z-index:1000;display:flex;gap:8px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}}
.dl-bar a{{display:inline-flex;align-items:center;gap:5px;padding:6px 11px;
  font-size:13px;font-weight:600;line-height:1;text-decoration:none;border-radius:6px;
  color:#fff;background:#2563eb;box-shadow:0 1px 3px rgba(0,0,0,.18);
  border:1px solid rgba(0,0,0,.06);transition:background .15s ease;}}
.dl-bar a:hover{{background:#1d4ed8;}}
.dl-bar a.docx{{background:#2b579a;}}
.dl-bar a.docx:hover{{background:#1f4478;}}
@media print{{.dl-bar{{display:none !important;}}}}
@media (max-width:640px){{.dl-bar{{position:static;justify-content:flex-end;margin:8px 10px 0;}}}}
</style>
<nav class="dl-bar" aria-label="Download this paper">
  <a class="pdf" href="paper.pdf" download title="Download PDF">PDF</a>
  <a class="docx" href="paper.docx" download title="Download Word (.docx)">DOCX</a>
</nav>
{MARKER_END}
"""


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/index.html")
    html = path.read_text(encoding="utf-8")
    # Drop any previously injected bar (idempotency).
    html = re.sub(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
                  "", html, flags=re.S).lstrip("\n")
    # Insert right after <body ...>
    m = re.search(r"<body[^>]*>", html, flags=re.I)
    if not m:
        print("no <body> tag found", file=sys.stderr)
        return 2
    idx = m.end()
    html = html[:idx] + "\n" + BAR + html[idx:]
    path.write_text(html, encoding="utf-8")
    print(f"Injected download bar into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
