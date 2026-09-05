from __future__ import annotations

import re
from pathlib import Path

UI_ROOT = Path(__file__).parents[2] / "ui"


def test_ui_is_framework_free_semantic_and_modular() -> None:
    index = (UI_ROOT / "index.html").read_text(encoding="utf-8")
    assert '<script type="module" src="/src/app.js">' in index
    assert all(
        f"<{element}" in index for element in ("header", "main", "section", "form", "footer")
    )
    assert "aria-live" in index
    assert "skip-link" in index
    assert not any(token in index.lower() for token in ("react", "vue", "angular"))
    assert (UI_ROOT / "styles" / "app.css").is_file()
    assert (UI_ROOT / "src" / "api.js").is_file()
    assert (UI_ROOT / "src" / "map.js").is_file()
    assert (UI_ROOT / "src" / "chart.js").is_file()


def test_only_api_client_constructs_versioned_endpoint_urls() -> None:
    modules = tuple((UI_ROOT / "src").glob("*.js"))
    api_module = UI_ROOT / "src" / "api.js"
    for module in modules:
        text = module.read_text(encoding="utf-8")
        if module != api_module:
            assert "/api/v1" not in text
            assert not re.search(r"fetch\s*\(", text)
    assert "/api/v1" in api_module.read_text(encoding="utf-8")


def test_static_files_do_not_embed_server_registry_inventories() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in UI_ROOT.rglob("*.*"))
    forbidden = (
        "google/gemma-4-",
        "max_weighted_coverage",
        "weighted_p_median",
        "cooling_center_coverage",
        'name: "improve"',
        'value="cuda"',
        'value="accelerate"',
    )
    assert not any(value in text for value in forbidden)


def test_responsive_styles_cover_narrow_and_wide_layouts() -> None:
    styles = (UI_ROOT / "styles" / "app.css").read_text(encoding="utf-8")
    assert "grid-template-columns" in styles
    assert "@media (max-width: 720px)" in styles
    assert ":focus-visible" in styles
    assert "prefers-reduced-motion" in styles


def test_primary_ui_requires_only_a_message_and_displays_the_answer() -> None:
    index = (UI_ROOT / "index.html").read_text(encoding="utf-8")
    app = (UI_ROOT / "src" / "app.js").read_text(encoding="utf-8")
    assert '<textarea id="message"' in index
    assert 'id="answer"' in index
    assert 'id="cancel-run"' in index
    assert all(name not in index for name in ("problem-example", "model-profile", "total-tokens"))
    assert "const request = { message };" in app
    assert "result.answer" in app
    assert "innerHTML" not in app
