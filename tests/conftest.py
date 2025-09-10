"""Test configuration and fixtures for mechafil-server tests."""

import json
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest
import requests
import httpx


@pytest.fixture(scope="session")
def live_server() -> Generator[str, None, None]:
    """Start real FastAPI server for testing.
    
    This fixture starts the actual server on a test port and waits for it
    to be ready before yielding the base URL. After tests complete,
    it cleans up the server process.
    """
    test_port = "8001"
    base_url = f"http://localhost:{test_port}"
    
    # Start server in background
    proc = subprocess.Popen([
        "poetry", "run", "uvicorn", 
        "mechafil_server.main:app",
        "--host", "localhost",
        "--port", test_port
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to be ready (max 60 seconds)
    server_ready = False
    for _ in range(60):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                server_ready = True
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(1)
    
    if not server_ready:
        proc.terminate()
        proc.wait()
        pytest.fail("Failed to start test server within 60 seconds")
    
    yield base_url
    
    # Cleanup
    proc.terminate()
    proc.wait()


@pytest.fixture
def api_client(live_server: str) -> httpx.Client:
    """HTTP client for making API calls to the live server."""
    return httpx.Client(base_url=live_server, timeout=30.0)


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Get the original test data directory."""
    return project_root / "test"


@pytest.fixture
def offline_simulation_scripts() -> dict:
    """Get script names for offline simulations in tests/ directory."""
    return {
        "historical_data": "test-data-fetching.py",
        "lock_target": "test-simulation.py", 
        "forecast_len": "test-simulation.py",
        "sector_duration": "test-simulation.py",
        "default_simulation": "test-simulation.py",
    }


def run_offline_simulation(script_path: Path, working_dir: Path, **kwargs) -> dict:
    """Run an offline simulation script and return the results.
    
    Args:
        script_path: Path to the simulation script
        working_dir: Directory to run the script in
        **kwargs: Additional command line arguments
        
    Returns:
        Dictionary containing the simulation results
    """
    cmd = ["poetry", "run", "python", str(script_path)]
    
    # Add command line arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    result = subprocess.run(
        cmd, 
        cwd=working_dir, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Offline simulation failed: {result.stderr}")
    
    # Look for the output JSON file
    output_files = ["offline_simulation.json", "spacescope_results.json"]
    for filename in output_files:
        output_file = working_dir / filename
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
    
    raise RuntimeError(f"No output file found in {working_dir}")