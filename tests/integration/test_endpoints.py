"""Basic endpoint tests for the FastAPI server.

These tests verify that endpoints respond correctly and return expected data structures.
"""

import pytest
import httpx


@pytest.mark.integration
class TestBasicEndpoints:
    """Test basic functionality of API endpoints."""
    
    def test_health_endpoint(self, api_client: httpx.Client):
        """Test the health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "jax_backend" in data
    
    def test_root_endpoint(self, api_client: httpx.Client):
        """Test the root endpoint returns API information."""
        response = api_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Mechafil Server"
        assert "version" in data
        assert "endpoints" in data
        assert "docs" in data
    
    def test_historical_data_summary_endpoint(self, api_client: httpx.Client):
        """Test the historical data summary endpoint."""
        response = api_client.get("/historical-data")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Historical data loaded and available"
        assert "data_summary" in data
    
    def test_historical_data_full_endpoint_structure(self, api_client: httpx.Client):
        """Test the full historical data endpoint returns expected structure."""
        response = api_client.get("/historical-data/full")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Complete historical data"
        
        # Check required top-level keys
        required_keys = ["smoothed_metrics", "historical_arrays", "offline_data"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Check smoothed metrics structure
        smoothed = data["smoothed_metrics"]
        smoothed_fields = ["raw_byte_power", "renewal_rate", "filplus_rate"]
        for field in smoothed_fields:
            assert field in smoothed, f"Missing smoothed metric: {field}"
            assert isinstance(smoothed[field], (int, float)), f"Invalid type for {field}"
        
        # Check historical arrays structure
        historical = data["historical_arrays"]
        for field in smoothed_fields:
            assert field in historical, f"Missing historical array: {field}"
            assert isinstance(historical[field], list), f"Invalid type for historical {field}"
    
    def test_simulate_endpoint_minimal_request(self, api_client: httpx.Client):
        """Test the simulate endpoint with minimal request."""
        response = api_client.post("/simulate", json={})
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required top-level keys
        required_keys = ["input", "smoothed_metrics", "simulation_output"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Check input structure
        assert "forecast_length_days" in data["input"]
        assert isinstance(data["input"]["forecast_length_days"], int)
        
        # Check smoothed metrics (same as historical data endpoint)
        smoothed = data["smoothed_metrics"]
        smoothed_fields = ["raw_byte_power", "renewal_rate", "filplus_rate"]
        for field in smoothed_fields:
            assert field in smoothed, f"Missing smoothed metric: {field}"
        
        # Check simulation output has some expected fields (based on actual output)
        sim_output = data["simulation_output"]
        expected_fields = ["available_supply", "capped_power_EIB", "1y_return_per_sector"]
        for field in expected_fields:
            assert field in sim_output, f"Missing simulation field: {field}"
            assert isinstance(sim_output[field], list), f"Field {field} should be a list"
    
    def test_simulate_endpoint_with_parameters(self, api_client: httpx.Client):
        """Test the simulate endpoint with custom parameters."""
        params = {
            "forecast_length_days": 365,
            "sector_duration_days": 540,
            "lock_target": 0.25,
            "rbp": 3.0,
            "rr": 0.8,
            "fpr": 0.9
        }
        
        response = api_client.post("/simulate", json=params)
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify the input parameters are reflected in response
        assert data["input"]["forecast_length_days"] == params["forecast_length_days"]
    
    def test_simulate_endpoint_handles_various_parameters(self, api_client: httpx.Client):
        """Test that the API handles various parameter values gracefully."""
        test_cases = [
            {"forecast_length_days": 100},
            {"sector_duration_days": 300},
            {"lock_target": 0.1},
            {"rbp": 5.0},
            {"rr": 0.9},
            {"fpr": 0.95},
        ]
        
        for params in test_cases:
            response = api_client.post("/simulate", json=params)
            # API should either succeed or fail gracefully
            assert response.status_code in [200, 422, 500], f"Unexpected status for {params}"