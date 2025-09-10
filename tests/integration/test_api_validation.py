"""Integration tests that validate API responses against offline simulations.

These tests replicate the functionality of the shell scripts in the test/ directory
but use pytest framework for better automation and CI/CD integration.
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest
import httpx

from tests.utils.simulation_helpers import (
    run_offline_simulation_with_params, 
    compare_simulation_results,
    save_api_response_for_comparison
)


@pytest.mark.integration
class TestAPIValidation:
    """Test that API responses match offline simulation results."""
    
    def test_historical_data_endpoint_matches_offline(
        self, 
        api_client: httpx.Client, 
        offline_simulation_scripts: Dict[str, str],
        tmp_path: Path
    ):
        """Test that /historical-data/full matches offline historical data simulation.
        
        """
        # Get API response
        response = api_client.get("/historical-data/full")
        assert response.status_code == 200
        api_data = response.json()
        
        # Save API response for comparison (mimicking shell script behavior)
        save_api_response_for_comparison(api_data, tmp_path / "api_results.json")
        
        # Run offline simulation
        script_name = offline_simulation_scripts["historical_data"]
        project_root = Path(__file__).parent.parent.parent
        offline_data = run_offline_simulation_with_params(
            script_name, tmp_path, 
            project_root=project_root,
            api_data=api_data
        )
        
        # Compare results
        assert compare_simulation_results(api_data, offline_data), \
            "API historical data does not match offline simulation results"
    
    def test_default_simulation_endpoint(
        self, 
        api_client: httpx.Client, 
        offline_simulation_scripts: Dict[str, str],
        tmp_path: Path
    ):
        """Test /simulate with default parameters matches offline simulation.
        
        """
        # Get API response with default parameters
        response = api_client.post("/simulate", json={})
        assert response.status_code == 200
        api_data = response.json()
        
        # Save API response for comparison
        save_api_response_for_comparison(api_data, tmp_path / "api_results.json")
        
        # Run offline simulation with default parameters
        script_name = offline_simulation_scripts["default_simulation"]
        project_root = Path(__file__).parent.parent.parent
        offline_data = run_offline_simulation_with_params(
            script_name, tmp_path, 
            project_root=project_root,
            api_data=api_data
        )
        
        # Compare results
        assert compare_simulation_results(api_data, offline_data), \
            "API default simulation does not match offline simulation results"
    
    def test_lock_target_simulation(
        self, 
        api_client: httpx.Client, 
        offline_simulation_scripts: Dict[str, str],
        tmp_path: Path
    ):
        """Test /simulate with custom lock_target matches offline simulation.
        
        """
        params = {
            "forecast_length_days": 365,
            "sector_duration_days": 365,
            "lock_target": 0.1
        }
        
        # Get API response
        response = api_client.post("/simulate", json=params)
        assert response.status_code == 200
        api_data = response.json()
        
        # Save API response for comparison
        save_api_response_for_comparison(api_data, tmp_path / "api_results.json")
        
        # Run offline simulation with same parameters
        script_name = offline_simulation_scripts["lock_target"]
        project_root = Path(__file__).parent.parent.parent
        offline_data = run_offline_simulation_with_params(
            script_name, tmp_path, params, 
            project_root=project_root,
            api_data=api_data
        )
        
        # Compare results
        assert compare_simulation_results(api_data, offline_data), \
            "API lock target simulation does not match offline simulation results"
    
    def test_forecast_length_simulation(
        self, 
        api_client: httpx.Client, 
        offline_simulation_scripts: Dict[str, str],
        tmp_path: Path
    ):
        """Test /simulate with custom forecast_length_days matches offline simulation.
        
        """
        params = {
            "forecast_length_days": 365
        }
        
        # Get API response
        response = api_client.post("/simulate", json=params)
        assert response.status_code == 200
        api_data = response.json()
        
        # Save API response for comparison
        save_api_response_for_comparison(api_data, tmp_path / "api_results.json")
        
        # Run offline simulation with same parameters
        script_name = offline_simulation_scripts["forecast_len"]
        project_root = Path(__file__).parent.parent.parent
        offline_data = run_offline_simulation_with_params(
            script_name, tmp_path, params,
            project_root=project_root,
            api_data=api_data
        )
        
        # Compare results
        assert compare_simulation_results(api_data, offline_data), \
            "API forecast length simulation does not match offline simulation results"
    
    def test_sector_duration_simulation(
        self, 
        api_client: httpx.Client, 
        offline_simulation_scripts: Dict[str, str],
        tmp_path: Path
    ):
        """Test /simulate with custom sector_duration_days matches offline simulation.
        
        """
        params = {
            "forecast_length_days": 365,
            "sector_duration_days": 365
        }
        
        # Get API response
        response = api_client.post("/simulate", json=params)
        assert response.status_code == 200
        api_data = response.json()
        
        # Save API response for comparison
        save_api_response_for_comparison(api_data, tmp_path / "api_results.json")
        
        # Run offline simulation with same parameters
        script_name = offline_simulation_scripts["sector_duration"]
        project_root = Path(__file__).parent.parent.parent
        offline_data = run_offline_simulation_with_params(
            script_name, tmp_path, params,
            project_root=project_root,
            api_data=api_data
        )
        
        # Compare results
        assert compare_simulation_results(api_data, offline_data), \
            "API sector duration simulation does not match offline simulation results"
    
    @pytest.mark.parametrize("test_case", [
        {"forecast_length_days": 365},
        {"forecast_length_days": 1000, "lock_target": 0.25},
        {"sector_duration_days": 540, "lock_target": 0.3},
        {"forecast_length_days": 180, "sector_duration_days": 365, "lock_target": 0.2},
    ])
    def test_simulation_parameter_scenarios(
        self, 
        api_client: httpx.Client, 
        offline_simulation_scripts: Dict[str, str],
        tmp_path: Path,
        test_case: Dict[str, Any]
    ):
        """Test various parameter combinations for simulation consistency.
        
        This provides additional test coverage beyond the original shell scripts.
        """
        # Get API response
        response = api_client.post("/simulate", json=test_case)
        assert response.status_code == 200
        api_data = response.json()
        
        # Verify response structure
        assert "input" in api_data
        assert "simulation_output" in api_data
        assert "smoothed_metrics" in api_data
        
        # Verify input parameters were applied
        if "forecast_length_days" in test_case:
            assert api_data["input"]["forecast_length_days"] == test_case["forecast_length_days"]
        
        # Verify simulation output contains expected fields
        sim_output = api_data["simulation_output"]
        expected_fields = ["available_supply", "capped_power_EIB", "1y_return_per_sector"]
        for field in expected_fields:
            assert field in sim_output, f"Missing field {field} in simulation output"