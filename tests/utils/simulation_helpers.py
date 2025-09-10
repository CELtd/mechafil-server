"""Utilities for running and comparing simulations."""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def run_offline_simulation_with_params(
    script_name: str,  # Now just the script name, not full path
    working_dir: Path, 
    params: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
    api_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run offline simulation script with specified parameters.
    
    Args:
        script_name: Name of the script to run ('test-simulation.py' or 'test-data-fetching.py')
        working_dir: Directory to run the script in
        params: Dictionary of parameters to pass to the script
        project_root: Project root directory (where pyproject.toml is located)
        api_data: API response data to save for comparison (if needed by script)
        
    Returns:
        Dictionary containing the simulation results
    """
    # Use project root as working directory for poetry commands
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    # Build path to test script in tests/ directory
    script_path = project_root / "tests" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Test script not found: {script_path}")
    
    # Some scripts expect an api_results.json file for comparison
    # Create it in the project root if api_data is provided
    if api_data is not None:
        api_results_path = project_root / "api_results.json"
        with open(api_results_path, 'w') as f:
            json.dump(api_data, f, indent=2)
    
    cmd = ["poetry", "run", "python", str(script_path)]
    
    # Convert parameters to command line arguments
    if params:
        for key, value in params.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    # Set environment to use the working directory for output files
    env = {"PYTHONPATH": str(project_root)}
    
    result = subprocess.run(
        cmd, 
        cwd=project_root,  # Run from project root for poetry
        capture_output=True, 
        text=True,
        env={**subprocess.os.environ, **env}
    )
    
    # Clean up api_results.json if we created it
    if api_data is not None:
        api_results_path = project_root / "api_results.json"
        if api_results_path.exists():
            api_results_path.unlink()
    
    if result.returncode != 0:
        raise RuntimeError(f"Offline simulation failed: {result.stderr}")
    
    # Look for the output JSON file in both working_dir and project_root
    output_files = ["offline_simulation.json", "spacescope_results.json"]
    search_dirs = [working_dir, project_root]
    
    for search_dir in search_dirs:
        for filename in output_files:
            output_file = search_dir / filename
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                # Clean up the file after reading
                output_file.unlink()
                return data
    
    raise RuntimeError(f"No output file found in {working_dir} or {project_root}")


def compare_simulation_results(
    api_result: Dict[str, Any], 
    offline_result: Dict[str, Any],
    tolerance: float = 1e-10
) -> bool:
    """Compare API and offline simulation results with tolerance for floating point differences.
    
    Args:
        api_result: Results from API call
        offline_result: Results from offline simulation
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        True if results match within tolerance
    """
    def compare_values(a: Any, b: Any) -> bool:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= tolerance
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(compare_values(x, y) for x, y in zip(a, b))
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(compare_values(a[k], b[k]) for k in a.keys())
        else:
            return a == b
    
    return compare_values(api_result, offline_result)


def save_api_response_for_comparison(response_data: Dict[str, Any], output_path: Path) -> None:
    """Save API response to JSON file for comparison with offline simulation.
    
    Args:
        response_data: API response data
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(response_data, f, indent=2)