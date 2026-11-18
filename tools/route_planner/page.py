"""The map page as one self-contained HTML string.

QtWebEngine in the packaged app did not load the page's local `file://` scripts and stylesheets (Leaflet never
ran, the map stayed blank), so the view gets the page with every local asset inlined instead.
"""

from __future__ import annotations

import re
from pathlib import Path

STYLESHEET_TAG = re.compile(r'<link rel="stylesheet" href="([^"]+)">')
SCRIPT_TAG = re.compile(r'<script src="([^":]+)"></script>')   # local paths only; no scheme
PAGE_FILE = "index.html"


def _script_text(source: str) -> str:
    # a literal "</script" inside inlined code would end the tag early
    return source.replace("</script", "<\\/script")


def inline_page(web_dir: Path) -> str:
    """`index.html` with its local stylesheets and scripts replaced by their contents."""
    html = (web_dir / PAGE_FILE).read_text(encoding="utf-8")
    html = STYLESHEET_TAG.sub(
        lambda m: f"<style>\n{(web_dir / m.group(1)).read_text(encoding='utf-8')}\n</style>", html)
    return SCRIPT_TAG.sub(
        lambda m: f"<script>\n{_script_text((web_dir / m.group(1)).read_text(encoding='utf-8'))}\n</script>", html)
