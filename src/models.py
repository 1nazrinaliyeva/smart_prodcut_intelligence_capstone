from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_joblib(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def artifact_status(model_dir: Path) -> dict[str, bool]:
    return {
        "tabular_mlp": (model_dir / "tabular_mlp.keras").exists(),
        "vision_transfer": (
            (model_dir / "vision_mobilenetv2_transfer.keras").exists()
            or (model_dir / "vision_transfer_head.keras").exists()
        ),
        "transformer": (model_dir / "distilbert_review_sentiment").exists(),
        "tabular_preprocessor": (model_dir / "tabular_mlp_preprocessor.joblib").exists(),
    }
