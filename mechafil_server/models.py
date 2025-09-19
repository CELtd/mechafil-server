"""Pydantic models for the Mechafil Server API."""

from datetime import date
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
try:
    # Pydantic v2
    from pydantic import ConfigDict
except Exception:
    ConfigDict = None  # type: ignore



class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="0.1.0", description="Service version")
    jax_backend: Optional[str] = Field(None, description="JAX backend in use")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class SimulationRequest(BaseModel):
    """Simulation request parameters.

    All fields are optional; defaults are derived from cached historical data
    and repo conventions (see tests).
    """

    rbp: Optional[Union[float, List[float]]] = Field(
        default=None, description="Raw byte power onboarding (PIB/day), constant or array"
    )
    rr: Optional[Union[float, List[float]]] = Field(
        default=None, description="Renewal rate (0..1), constant or array"
    )
    fpr: Optional[Union[float, List[float]]] = Field(
        default=None, description="FIL+ rate (0..1), constant or array"
    )
    lock_target: Optional[Union[float, List[float]]] = Field(
        default=None, description="Target lock ratio (e.g., 0.3), float or array"
    )
    forecast_length_days: Optional[int] = Field(
        default=None, description="Forecast length in days"
    )
    sector_duration_days: Optional[int] = Field(
        default=None, description="Average sector duration in days"
    )
    output: Optional[Union[str, List[str]]] = Field(
        default=None, description="Specific output field(s) to return. If not specified, returns all fields."
    )
    
    @field_validator('output')
    @classmethod
    def validate_output_fields(cls, v):
        """Validate that requested output fields are valid simulation output fields."""
        if v is None:
            return v
            
        # Valid simulation output fields
        valid_fields = {
            "1y_return_per_sector", "1y_sector_roi", "available_supply", "capped_power_EIB",
            "circ_supply", "cum_baseline_reward", "cum_capped_power_EIB", "cum_network_reward",
            "cum_simple_reward", "day_locked_pledge", "day_network_reward", "day_onboarded_power_QAP_PIB",
            "day_pledge_per_QAP", "day_renewed_pledge", "day_renewed_power_QAP_PIB", "day_rewards_per_sector",
            "days", "disbursed_reserve", "full_renewal_rate", "network_QAP_EIB", "network_RBP_EIB",
            "network_baseline_EIB", "network_gas_burn", "network_locked", "network_locked_pledge",
            "network_locked_reward", "network_time", "one_year_vest_saft", "qa_day_onboarded_power_pib",
            "qa_day_renewed_power_pib", "qa_sched_expire_power_pib", "qa_total_power_eib",
            "rb_day_onboarded_power_pib", "rb_day_renewed_power_pib", "rb_sched_expire_power_pib",
            "rb_total_power_eib", "six_month_vest_saft", "six_year_vest_foundation", "six_year_vest_pl",
            "six_year_vest_saft", "three_year_vest_saft", "total_day_vest", "total_vest", "two_year_vest_saft"
        }
        
        # Convert single string to list for uniform processing
        fields_to_check = [v] if isinstance(v, str) else v
        
        # Check if all requested fields are valid
        invalid_fields = [field for field in fields_to_check if field not in valid_fields]
        if invalid_fields:
            raise ValueError(f"Invalid output field(s): {invalid_fields}. Valid fields: {sorted(valid_fields)}")
        
        return v

    # Example payload shown in Swagger UI for quick testing
    if ConfigDict is not None:
        model_config = ConfigDict(
            json_schema_extra={
                "example": {
                    "rbp": 3.3795318603515625,
                    "rr": 0.834245140193526,
                    "fpr": 0.8558804137732767,
                    "lock_target": 0.3,
                    "forecast_length_days": 365,
                    "sector_duration_days": 540,
                    "output": ["available_supply", "network_RBP_EIB"]
                }
            }
        )

class SimulationError(BaseModel):
    status: str = Field(default="error")
    message: str
    detail: Optional[Dict[str, Any]] = None
