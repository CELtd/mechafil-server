#!/usr/bin/env python3
"""
Build documentation automatically during installation.
This script is called by Poetry after dependencies are installed.
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Build Sphinx documentation."""
    # Get the project root (parent of mechafil_server directory)
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        print("Warning: docs directory not found, skipping documentation build")
        return 0

    print("Building documentation...")

    try:
        # Change to docs directory and run make html
        result = subprocess.run(
            ["make", "html"],
            cwd=docs_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print("Documentation built successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to build documentation: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        # Don't fail the installation if docs build fails
        return 0
    except FileNotFoundError:
        print("Warning: 'make' command not found, skipping documentation build", file=sys.stderr)
        return 0


if __name__ == "__main__":
    sys.exit(main())
