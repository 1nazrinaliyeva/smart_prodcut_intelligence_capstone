from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root from either notebooks, app, or src."""
    current = Path.cwd().resolve()
    if current.name in {"notebooks", "app", "src"}:
        return current.parent
    return current


def existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def format_missing(path: Path) -> str:
    return f"Missing artifact: {path}"
