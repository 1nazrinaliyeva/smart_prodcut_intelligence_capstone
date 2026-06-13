from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Smart Product Intelligence Streamlit app."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Streamlit in headless mode.",
    )
    default_project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing data/, reports/, and models/. Defaults to this repository.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing processed parquet/csv files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory containing milestone report CSV/JSON files.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing trained model artifacts.",
    )
    args = parser.parse_args()

    project_root = (args.project_root or default_project_root).resolve()
    os.environ["SMART_PRODUCT_PROJECT_ROOT"] = str(project_root)
    if args.processed_dir is not None:
        os.environ["SMART_PRODUCT_PROCESSED_DIR"] = str(args.processed_dir.resolve())
    if args.reports_dir is not None:
        os.environ["SMART_PRODUCT_REPORTS_DIR"] = str(args.reports_dir.resolve())
    if args.models_dir is not None:
        os.environ["SMART_PRODUCT_MODELS_DIR"] = str(args.models_dir.resolve())

    app_path = Path(__file__).with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        str(args.headless).lower(),
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
