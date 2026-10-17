"""Smoke test: the dashboard renders from the committed processed data."""
import pytest

from app import app
from nbafinals.config import OUT_DIR

pytestmark = pytest.mark.skipif(not (OUT_DIR / "current.json").exists(), reason="run the build first")


@pytest.mark.parametrize("url", ["/", "/team/OKC", "/report"])
def test_pages_render(url):
    resp = app.test_client().get(url)
    assert resp.status_code == 200
    assert b"<canvas" in resp.data


def test_unknown_team_is_404():
    assert app.test_client().get("/team/XYZ").status_code == 404


def test_published_title_odds_sum_to_100_percent():
    import json
    teams = json.loads((OUT_DIR / "current.json").read_text())["teams"]
    total = sum(t["odds"]["champion"] for t in teams if t["odds"])
    assert total == pytest.approx(1.0, abs=1e-3)
