"""Main FastAPI application for the Mechafil Server."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import jax
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from mechafil_jax import sim as mechafil_sim
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True) 

from .models import (
    HealthResponse,
    ErrorResponse,
    SimulationRequest,
    SimulationError,
)
from .data import Data
from .config import settings
from .scheduler import DataRefreshScheduler

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

# Global data handler and scheduler
loaded_data: Data | None = None
data_scheduler: DataRefreshScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global loaded_data, data_scheduler

    # Startup
    logger.info("Starting up Mechafil Server...")
    logger.info(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    logger.info(f"JAX devices: {jax.devices()}")

    try:
        loaded_data = Data()
        loaded_data.load_historical_data()
        logger.info("Historical data loaded successfully")
        
        # Start the data refresh scheduler
        data_scheduler = DataRefreshScheduler(loaded_data.refresh_historical_data)
        data_scheduler.start()
        logger.info(f"Data refresh scheduler started. Daily refresh at {settings.RELOAD_TRIGGER} UTC")
        
    except Exception as e:
        logger.error(f"Failed to load historical data on startup: {e}")
        logger.warning("Server will continue without historical data")

    yield

    # Shutdown
    logger.info("Shutting down Mechafil Server...")
    if data_scheduler:
        try:
            await data_scheduler.stop_async()
        except Exception as e:
            logger.warning(f"Error stopping scheduler: {e}")
            # Fallback to sync stop
            try:
                data_scheduler.stop()
            except Exception as e2:
                logger.warning(f"Error with fallback stop: {e2}")


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


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Mechafil Server",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health (GET) - Server health check and JAX backend info",
            "historical_data": "/historical-data (GET) - Historical data summary",
            "historical_data_full": "/historical-data/full (GET) - Full historical data with arrays",
            "simulate": "/simulate (POST) - Run Filecoin forecast simulation",
        },
        "quick_test": "curl -X POST http://localhost:8000/simulate -H 'Content-Type: application/json' -d '{}'",
        "template_info": "Empty request '{}' uses defaults from historical data",
    }


@app.get("/historical-data", tags=["Data"])
async def get_historical_data():
    """Get pretty printed historical data if available."""
    global loaded_data

    if loaded_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data handler not initialized",
        )

    try:
        summary = loaded_data.get_historical_data_summary()
        return {
            "message": "Historical data loaded and available",
            "data_summary": summary,
        }
    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical data: {str(e)}",
        )


@app.get("/historical-data/full", tags=["Data"])
async def get_historical_data_full():
    """Get complete historical data with all values."""
    global loaded_data

    if loaded_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data handler not initialized",
        )

    try:
        logger.info("Getting full historical data...")
        historical_data = loaded_data.get_historical_data()
        if historical_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No historical data available",
            )

        hist_data = loaded_data.get_historical_data()
        if not hist_data:
            raise RuntimeError("No historical data loaded")
        
        offline_data = hist_data["offline_data"]
        hist_rbp = hist_data["hist_rbp"]
        hist_rr = hist_data["hist_rr"]
        hist_fpr = hist_data["hist_fpr"]
        
        smoothed_rbp = hist_data["smoothed_rbp"]
        smoothed_rr = hist_data["smoothed_rr"]
        smoothed_fpr = hist_data["smoothed_fpr"]


        return {
            "message": "Complete historical data",
            "smoothed_metrics": {
                "raw_byte_power": float(smoothed_rbp),
                "renewal_rate": float(smoothed_rr),
                "filplus_rate": float(smoothed_fpr),
            },
            "historical_arrays": {
                "raw_byte_power": [float(x) for x in hist_rbp],
                "renewal_rate": [float(x) for x in hist_rr],
                "filplus_rate": [float(x) for x in hist_fpr],
            },
            "offline_data": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in offline_data.items()},
        }
    except Exception as e:
        logger.error(f"Error retrieving full historical data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical data: {str(e)}",
        )


@app.post("/simulate", tags=["Simulation"])
async def simulate(req: SimulationRequest):
    """
    Run a Filecoin forecast simulation.

    Example curl:
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 3650, "lock_target": 0.3}'
    """
    global loaded_data

    if loaded_data is None or loaded_data.historical_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical data not loaded yet; try again shortly"
        )

    # Unpack request data with defaults from historical data
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

    try:
        #offline_data = hist_data["offline_data"]
        start_date = hist_data["start_date"]
        current_date = hist_data["current_date"]
        simulation_offline_data = loaded_data.trim_data_for_simulation(forecast_len)

        # Convert parameters to JAX arrays (handle both constants and arrays)
        if isinstance(rbp_value, list):
            rbp = jnp.array(rbp_value)
        else:
            rbp = jnp.ones(forecast_len) * rbp_value
            
        if isinstance(rr_value, list):
            rr = jnp.array(rr_value)
        else:
            rr = jnp.ones(forecast_len) * rr_value
            
        if isinstance(fpr_value, list):
            fpr = jnp.array(fpr_value)
        else:
            fpr = jnp.ones(forecast_len) * fpr_value

        results = mechafil_sim.run_sim(
            rbp, rr, fpr, lock_target, start_date, current_date,
            forecast_len, sector_duration_days, simulation_offline_data,
            use_available_supply=False
        )

        return {
            "input": {
                "forecast_length_days": forecast_len
            },
            "smoothed_metrics": {
                "raw_byte_power": float(smoothed_rbp),
                "renewal_rate": float(smoothed_rr),
                "filplus_rate": float(smoothed_fpr),
            },
            "simulation_output": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in results.items()
            },
        }

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
        "mechafil_server.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
