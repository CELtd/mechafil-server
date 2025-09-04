"""Pydantic models for the Mechafil Server API."""

from datetime import date
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator



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

class SimulationError(BaseModel):
    status: str = Field(default="error")
    message: str
    detail: Optional[Dict[str, Any]] = None
