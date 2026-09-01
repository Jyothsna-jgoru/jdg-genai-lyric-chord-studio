from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generation_progress_ui_is_wired_and_accessible():
    html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    styles = (ROOT / "app" / "static" / "styles.css").read_text(encoding="utf-8")
    assert 'id="generation-progress"' in html
    assert 'role="status"' in html and 'aria-live="polite"' in html
    assert "Local CPU generation takes about 1 minute and may need up to 90 seconds." in html
    assert "beginGeneration(\"song\")" in script and "finally { endGeneration(); }" in script
    assert "generation-spinner" in styles and "prefers-reduced-motion" in styles
