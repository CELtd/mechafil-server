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
            "historical_data": "/historical-data (GET) - Historical data with configurable averaging",
            "historical_data_full": "/historical-data/full (GET) - Full historical data with arrays",
            "simulate": "/simulate (POST) - Run Filecoin forecast simulation with weekly averaged results (supports 'output' field filtering)",
            "simulate_full": "/simulate/full (POST) - Run Filecoin forecast simulation with full detailed results",
        },
        "quick_test": "curl -X POST http://localhost:8000/simulate -H 'Content-Type: application/json' -d '{}' (averaged) or /simulate/full (detailed)",
        "template_info": "Empty request '{}' uses defaults from historical data",
    }


@app.get("/historical-data", tags=["Data"])
async def get_historical_data():
    """Get historical data with configurable averaging (weekly or 10-day windows)."""
    global loaded_data

    if loaded_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data handler not initialized",
        )

    try:
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

        # Calculate averaged data
        def calculate_averaged_data(data_array):
            """Calculate weekly or rolling average for an array based on config."""
            import numpy as np
            from datetime import datetime, timedelta
            
            data = np.array(data_array)
            
            if not settings.USE_WEEKLY_AVERAGING:
                # Fallback to 10-day rolling average
                window_size = 10
                if len(data) < window_size:
                    return [round(float(np.mean(data)), 2)]
                
                averaged = []
                for i in range(0, len(data), window_size):
                    window = data[i:i + window_size]
                    averaged.append(round(float(np.mean(window)), 2))
                return averaged
            
            # Weekly averaging (Monday to Sunday)
            if len(data) < 7:
                return [round(float(np.mean(data)), 2)]
            
            # Assume data starts from the start_date and goes sequentially
            start_date = hist_data.get("start_date")
            if not start_date:
                # Fallback to simple 7-day windows if no start date
                averaged = []
                for i in range(0, len(data), 7):
                    window = data[i:i + 7]
                    averaged.append(round(float(np.mean(window)), 2))
                return averaged
            
            # Find the first Monday
            current_date = start_date
            days_until_monday = (7 - current_date.weekday()) % 7
            if days_until_monday == 0 and current_date.weekday() != 0:
                days_until_monday = 7
            
            first_monday_index = days_until_monday
            
            averaged = []
            
            # Handle partial week before first Monday
            if first_monday_index > 0:
                partial_week = data[:first_monday_index]
                if len(partial_week) > 0:
                    averaged.append(round(float(np.mean(partial_week)), 2))
            
            # Process complete weeks (Monday to Sunday)
            for i in range(first_monday_index, len(data), 7):
                week_data = data[i:i + 7]
                if len(week_data) > 0:
                    averaged.append(round(float(np.mean(week_data)), 2))
            
            return averaged

        averaging_method = "weekly (Monday-Sunday)" if settings.USE_WEEKLY_AVERAGING else "10-day windows"
        
        return {
            "message": f"Historical data averaged over {averaging_method}",
            "averaging_method": averaging_method,
            "smoothed_metrics": {
                "raw_byte_power": round(float(smoothed_rbp), 2),
                "renewal_rate": round(float(smoothed_rr), 2),
                "filplus_rate": round(float(smoothed_fpr), 2),
            },
            "averaged_arrays": {
                "raw_byte_power": calculate_averaged_data(hist_rbp),
                "renewal_rate": calculate_averaged_data(hist_rr),
                "filplus_rate": calculate_averaged_data(hist_fpr),
            },
            "offline_data_averaged": {
                k: (calculate_averaged_data(v) if hasattr(v, "__iter__") and not isinstance(v, str) and len(v) > 1 else [round(float(v), 2)] if isinstance(v, (int, float)) else v) 
                for k, v in offline_data.items()
            },
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
                "raw_byte_power": round(float(smoothed_rbp), 2),
                "renewal_rate": round(float(smoothed_rr), 2),
                "filplus_rate": round(float(smoothed_fpr), 2),
            },
            "historical_arrays": {
                "raw_byte_power": [round(float(x), 2) for x in hist_rbp],
                "renewal_rate": [round(float(x), 2) for x in hist_rr],
                "filplus_rate": [round(float(x), 2) for x in hist_fpr],
            },
            "offline_data": {k: ([round(float(item), 2) for item in v] if hasattr(v, "__iter__") and not isinstance(v, str) else round(float(v), 2) if isinstance(v, (int, float)) else v) for k, v in offline_data.items()},
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

        # Apply weekly averaging to simulation results
        def calculate_weekly_average_results(data_array):
            """Calculate weekly average for simulation result arrays."""
            import numpy as np
            
            data = np.array(data_array)
            if len(data) < 7:
                return [round(float(np.mean(data)), 2)]
            
            averaged = []
            for i in range(0, len(data), 7):
                week_data = data[i:i + 7]
                if len(week_data) > 0:
                    averaged.append(round(float(np.mean(week_data)), 2))
            
            return averaged

        # Apply weekly averaging to all result arrays
        averaged_results = {}
        for k, v in results.items():
            if hasattr(v, "__iter__") and not isinstance(v, str) and len(v) > 1:
                averaged_results[k] = calculate_weekly_average_results(v)
            elif isinstance(v, (int, float)):
                averaged_results[k] = round(float(v), 2)
            else:
                averaged_results[k] = v

        # Filter results based on output parameter
        if req.output is not None:
            # Convert single string to list for uniform processing
            requested_fields = [req.output] if isinstance(req.output, str) else req.output
            
            # Filter simulation_output to only include requested fields
            filtered_results = {
                field: averaged_results.get(field)
                for field in requested_fields
                if field in averaged_results
            }
            
            # Check if any requested fields were not found
            missing_fields = [field for field in requested_fields if field not in averaged_results]
            if missing_fields:
                logger.warning(f"Requested fields not found in simulation results: {missing_fields}")
            
            simulation_output = filtered_results
        else:
            simulation_output = averaged_results

        return {
            "input": {
                "forecast_length_days": forecast_len
            },
            "smoothed_metrics": {
                "raw_byte_power": round(float(smoothed_rbp), 2),
                "renewal_rate": round(float(smoothed_rr), 2),
                "filplus_rate": round(float(smoothed_fpr), 2),
            },
            "averaging_method": "weekly (7-day windows)",
            "simulation_output": simulation_output,
        }

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/simulate/full", tags=["Simulation"])
async def simulate_full(req: SimulationRequest):
    """
    Run a Filecoin forecast simulation with full detailed results.

    Example curl:
      curl -X POST http://localhost:8000/simulate/full \
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
                "raw_byte_power": round(float(smoothed_rbp), 2),
                "renewal_rate": round(float(smoothed_rr), 2),
                "filplus_rate": round(float(smoothed_fpr), 2),
            },
            "simulation_output": {
                k: ([round(float(item), 2) for item in v] if hasattr(v, "__iter__") and not isinstance(v, str) else round(float(v), 2) if isinstance(v, (int, float)) else v)
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
