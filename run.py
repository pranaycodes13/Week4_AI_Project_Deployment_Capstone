"""Main application entry point for the Wine Classification & Prediction service.

Supports pipeline execution, model training, evaluation, test suite running, and API server launching.
"""

import argparse
import os
import subprocess
import sys

import uvicorn

from src.config import PROJECT_ROOT, ensure_directories


def main() -> None:
    """Entry point for project execution and service orchestration."""
    ensure_directories()

    parser = argparse.ArgumentParser(
        description="Wine Classification & Prediction CLI & Server Runner"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "serve", "ui"],
        help="Command to execute: 'status' (default), 'serve' (launch FastAPI server), or 'ui' (launch Streamlit dashboard).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind socket host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind socket port (default: 8000 for serve, 8501 for ui)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code change (serve command only)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        port = args.port or int(os.getenv("PORT", "8000"))
        host = args.host if args.host != "127.0.0.1" else os.getenv("HOST", "127.0.0.1")
        print(f"Starting FastAPI Wine Classification API server at http://{host}:{port}")
        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=args.reload,
        )
        return

    if args.command == "ui":
        ui_port = args.port or 8501
        frontend_script = PROJECT_ROOT / "src" / "frontend" / "app.py"
        print(f"Launching Streamlit Frontend Dashboard at http://{args.host}:{ui_port}")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(frontend_script),
            "--server.address",
            args.host,
            "--server.port",
            str(ui_port),
        ]
        subprocess.run(cmd, check=False)
        return

    print("=" * 60)
    print("Wine Classification & Prediction API")
    print(f"Project Root: {PROJECT_ROOT}")
    print("Current Phase: Phase 8 (Streamlit Frontend Dashboard & FastAPI Integration)")
    print("=" * 60)
    print("\nAvailable execution commands:")
    print("  python -m src.data.loader            - Ingest raw dataset & inspect")
    print("  python -m src.preprocessing.pipeline - Run stratified split & save processed data")
    print("  python -m src.training.train         - Run 5-fold CV & model selection (Phase 3)")
    print("  python -m src.evaluation.evaluate    - Evaluate holdout test set & serialize (Phase 4)")
    print("  python run.py serve [--port 8000]    - Launch FastAPI prediction service (Phase 5)")
    print("  python run.py ui [--port 8501]       - Launch Streamlit frontend dashboard (Phase 8)")
    print("  uvicorn src.api.app:app --reload     - Run Uvicorn development server")
    print("  pytest -v                            - Execute automated test suite")
    print("  ruff check .                         - Run static analysis / linting")
    print("=" * 60)


if __name__ == "__main__":
    main()

