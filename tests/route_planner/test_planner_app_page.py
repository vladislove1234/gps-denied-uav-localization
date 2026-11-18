from __future__ import annotations

from tools.route_planner.main_window import WEB_DIR
from tools.route_planner.page import inline_page


def test_page_has_no_local_asset_references() -> None:
    html = inline_page(WEB_DIR)
    assert 'src="vendor/' not in html and 'href="vendor/' not in html and 'src="app.js"' not in html
    assert "L.map(" in html and "new QWebChannel" in html and "#map" in html
