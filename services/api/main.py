"""Main FastAPI application for the Mechafil Server."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import date, timedelta

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True) 

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from mechafil_jax import sim as mechafil_sim

from .models import (
    HealthResponse,
    ErrorResponse,
    SimulationRequest,
    SimulationError,
)
from .data import Data
from .config import settings
from .results import SimulationResults, FetchDataResults

# Load environment variables from common locations
load_dotenv()
try:
    here = Path(__file__).resolve()
    server_dir = here.parent
    repo_root = server_dir.parent
    for env_path in [repo_root / ".env", server_dir / ".env", repo_root / ".test-env"]:
        if env_path.exists():
            load_dotenv(env_path)
except Exception:
    pass

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global data handler
loaded_data: Data | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global loaded_data

    # Startup
    logger.info("Starting up Mechafil Server...")
    logger.info(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    logger.info(f"JAX devices: {jax.devices()}")

    try:
        loaded_data = Data()
        loaded_data.load_historical_data()
        logger.info("Historical data loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load historical data on startup: {e}")
        logger.warning("Server will continue without historical data")

    yield

    # Shutdown
    logger.info("Shutting down Mechafil Server...")


# Create FastAPI app
app = FastAPI(
    title="Mechafil Server",
    description="FastAPI server for running mechafil-jax simulations",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for Read the Docs documentation
docs_path = Path(__file__).parent.parent / "docs" / "build" / "html"
if docs_path.exists():
    app.mount("/documentation", StaticFiles(directory=str(docs_path), html=True), name="documentation")
    logger.info(f"Mounted Read the Docs documentation at /documentation")
else:
    logger.warning(f"Documentation path not found: {docs_path}. Build docs with: cd docs && poetry run make html")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        jax_backend = jax.lib.xla_bridge.get_backend().platform
        return HealthResponse(status="healthy", version="0.1.0", jax_backend=jax_backend)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy",
        )


@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """Redirect root to Read the Docs documentation."""
    # Check if docs are built, otherwise redirect to Swagger UI
    if docs_path.exists():
        return RedirectResponse(url="/documentation/index.html")
    else:
        return RedirectResponse(url="/docs")

@app.post("/admin/update-cache", tags=["Admin"])
async def update_cache():
    """
    Admin endpoint to trigger cache update from external source.

    This endpoint should be called by GitHub Actions or other trusted external triggers
    to update the shared cache with latest data from Spacescope.

    Returns:
        Status message indicating whether cache update was successful
    """
    global loaded_data

    logger.info("Admin endpoint: Cache update triggered")

    try:
        # Import cache updater logic
        from services.cache_updater.main import run_once
        import asyncio

        # Run the cache update (it's an async function)
        logger.info("Starting cache update...")
        exit_code = await run_once()

        if exit_code != 0:
            raise RuntimeError("Cache update returned non-zero exit code")

        logger.info("Cache update completed successfully")

        # Reload data in the API service
        if loaded_data is None:
            loaded_data = Data()

        logger.info("Reloading historical data from updated cache...")
        loaded_data.load_historical_data()
        logger.info("Historical data reloaded successfully")

        return {
            "status": "success",
            "message": "Cache updated and historical data reloaded",
            "timestamp": date.today().isoformat()
        }

    except Exception as e:
        logger.error(f"Cache update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache update failed: {str(e)}",
        )


@app.get("/historical-data", tags=["Data"])
async def get_historical_data_full():
    """Get historical data downsampled to Mondays for visualization."""
    global loaded_data

    if loaded_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data handler not initialized",
        )

    try:
        logger.info("Getting historical data (downsampled to Mondays)...")

        hist_data = loaded_data.get_historical_data()
        if not hist_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No historical data available",
            )

        offline_data = hist_data["offline_data"]
        hist_rbp = hist_data["hist_rbp"]
        hist_rr = hist_data["hist_rr"]
        hist_fpr = hist_data["hist_fpr"]

        smoothed_rbp = hist_data["smoothed_rbp"]
        smoothed_rr = hist_data["smoothed_rr"]
        smoothed_fpr = hist_data["smoothed_fpr"]

        start_date = hist_data["start_date"]

        # Wrap into FetchDataResults
        results = FetchDataResults.from_raw(
            hist_arrays={
                "raw_byte_power": hist_rbp,
                "renewal_rate": hist_rr,
                "filplus_rate": hist_fpr,
            },
            offline_data=offline_data,
            smoothed_rbp=smoothed_rbp,
            smoothed_rr=smoothed_rr,
            smoothed_fpr=smoothed_fpr,
        )

        # Downsample to Mondays
        results = results.downsample_mondays(start_date)

        return results.to_dict()

    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical data: {str(e)}",
        )

@app.post("/simulate", tags=["Simulation"])
async def simulate(req: SimulationRequest):
    """
    Run a Filecoin forecast simulation with weekly averaged results.

    Example curl commands:
      # Get all simulation results
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 365, "lock_target": 0.3}'
      
      # Get only specific output field
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 365, "output": "available_supply"}'
      
      # Get multiple specific output fields
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 365, "output": ["available_supply", "network_RBP_EIB"]}'
    """
    global loaded_data

    if loaded_data is None or loaded_data.historical_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical data not loaded yet; try again shortly"
        )

    # Get full simulation results first
    try:
        # Run the full simulation using the same logic as simulate_full
        hist_data = loaded_data.get_historical_data()
        if not hist_data:
            raise RuntimeError("No historical data loaded")

        # Use request values or fall back to historical data defaults
        forecast_len = req.forecast_length_days if req.forecast_length_days is not None else settings.WINDOW_DAYS
        sector_duration_days = req.sector_duration_days if req.sector_duration_days is not None else settings.SECTOR_DURATION_DAYS
        
        # Default values from smoothed historical data
        smoothed_rbp = hist_data["smoothed_rbp"]
        smoothed_rr = hist_data["smoothed_rr"] 
        smoothed_fpr = hist_data["smoothed_fpr"]
        
        # Use request parameters or defaults
        rbp_value = req.rbp if req.rbp is not None else smoothed_rbp
        rr_value = req.rr if req.rr is not None else smoothed_rr
        fpr_value = req.fpr if req.fpr is not None else smoothed_fpr
        lock_target = req.lock_target if req.lock_target is not None else settings.LOCK_TARGET

        start_date = hist_data["start_date"]
        current_date = hist_data["current_date"]
        sim_len = settings.WINDOW_DAYS

        # Convert parameters to JAX arrays (handle both constants and arrays)
        if isinstance(rbp_value, list):
            rbp = jnp.array(rbp_value)
        else:
            rbp = jnp.ones(sim_len) * rbp_value
            
        if isinstance(rr_value, list):
            rr = jnp.array(rr_value)
        else:
            rr = jnp.ones(sim_len) * rr_value
            
        if isinstance(fpr_value, list):
            fpr = jnp.array(fpr_value)
        else:
            fpr = jnp.ones(sim_len) * fpr_value

        raw_results = mechafil_sim.run_sim(
            rbp, rr, fpr, lock_target, start_date, current_date,
            sim_len, sector_duration_days, hist_data["offline_data"],
            use_available_supply=False
        )
        results = SimulationResults.from_raw(
            raw_results, start_date, current_date, forecast_len,
            smoothed_rbp, smoothed_rr, smoothed_fpr
        )

        # Downsample to Mondays
        results = results.downsample_mondays(start_date)
        
        # Filter output if requested
        if req.output is not None:
            results = results.filter_fields(req.output)
        
        return results.to_dict() 

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.post("/simulate/full", tags=["Simulation"])
async def simulatefull(req: SimulationRequest):
    """
    Run a Filecoin forecast simulation with weekly averaged results.

    Example curl commands:
      # Get all simulation results
      curl -X POST http://localhost:8000/simulate/full \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 365, "lock_target": 0.3}'
      
      # Get only specific output field
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 365, "output": "available_supply"}'
      
      # Get multiple specific output fields
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 365, "output": ["available_supply", "network_RBP_EIB"]}'
    """
    global loaded_data

    if loaded_data is None or loaded_data.historical_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical data not loaded yet; try again shortly"
        )

    # Get full simulation results first
    try:
        # Run the full simulation using the same logic as simulate_full
        hist_data = loaded_data.get_historical_data()
        if not hist_data:
            raise RuntimeError("No historical data loaded")

        # Use request values or fall back to historical data defaults
        forecast_len = req.forecast_length_days if req.forecast_length_days is not None else settings.WINDOW_DAYS
        sector_duration_days = req.sector_duration_days if req.sector_duration_days is not None else settings.SECTOR_DURATION_DAYS
        
        # Default values from smoothed historical data
        smoothed_rbp = hist_data["smoothed_rbp"]
        smoothed_rr = hist_data["smoothed_rr"] 
        smoothed_fpr = hist_data["smoothed_fpr"]
        
        # Use request parameters or defaults
        rbp_value = req.rbp if req.rbp is not None else smoothed_rbp
        rr_value = req.rr if req.rr is not None else smoothed_rr
        fpr_value = req.fpr if req.fpr is not None else smoothed_fpr
        lock_target = req.lock_target if req.lock_target is not None else settings.LOCK_TARGET

        start_date = hist_data["start_date"]
        current_date = hist_data["current_date"]
        sim_len = settings.WINDOW_DAYS 

        # Convert parameters to JAX arrays (handle both constants and arrays)
        if isinstance(rbp_value, list):
            rbp = jnp.array(rbp_value)
        else:
            rbp = jnp.ones(sim_len) * rbp_value
            
        if isinstance(rr_value, list):
            rr = jnp.array(rr_value)
        else:
            rr = jnp.ones(sim_len) * rr_value
            
        if isinstance(fpr_value, list):
            fpr = jnp.array(fpr_value)
        else:
            fpr = jnp.ones(sim_len) * fpr_value

        raw_results = mechafil_sim.run_sim(
            rbp, rr, fpr, lock_target, start_date, current_date,
            sim_len, sector_duration_days, hist_data["offline_data"],
            use_available_supply=False
        )
        results = SimulationResults.from_raw(
            raw_results, start_date, current_date, forecast_len,
            smoothed_rbp, smoothed_rr, smoothed_fpr
        )
        return results.to_dict() 

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")



def main():
    """Entry point for running the server."""
    import uvicorn

    host = settings.HOST
    port = settings.PORT
    reload = settings.RELOAD

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "services.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
