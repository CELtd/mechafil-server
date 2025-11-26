"""Mechafil Server - FastAPI server for running mechafil-jax simulations."""

__version__ = "0.1.0"


def _build_docs_on_import():
    """Build documentation on first import if not already built."""
    import os
    from pathlib import Path

    # Get the project root
    project_root = Path(__file__).parent.parent
    docs_build_dir = project_root / "docs" / "build" / "html"

    # Check if docs are already built
    if not (docs_build_dir / "index.html").exists():
        try:
            from .build_docs import main as build_docs_main
            build_docs_main()
        except Exception as e:
            # Silently fail - don't break imports if docs build fails
            pass


# Automatically build docs on first import
_build_docs_on_import()