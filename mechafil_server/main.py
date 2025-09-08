"""Main FastAPI application for the Mechafil Server."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import jax
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .models import (
    HealthResponse,
    ErrorResponse,
    SimulationRequest,
    SimulationError,
)
from .data import Data

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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global data handler
data: Data | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global data

    # Startup
    logger.info("Starting up Mechafil Server...")
    logger.info(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    logger.info(f"JAX devices: {jax.devices()}")

    try:
        data = Data()
        data.load_historical_data()
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
    allow_origins=["*"],  # TODO: restrict in production
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
        "health": "/health",
        "endpoints": {
            "simulate": "/simulate (POST) - Simulation with default values",
        },
        "template_info": "Empty request '{}' uses defaults from template-request.json",
        "quick_test": "curl -X POST http://localhost:8000/simulate -H 'Content-Type: application/json' -d '{}'",
        "historical_data": "/historical-data (GET) - View loaded historical data summary",
        "historical_data_full": "/historical-data/full (GET) - View all historical data values",
    }


@app.get("/historical-data", tags=["Data"])
async def get_historical_data():
    """Get pretty printed historical data if available."""
    global data

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data handler not initialized",
        )

    try:
        summary = data.get_historical_data_summary()
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
    global data

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data handler not initialized",
        )

    try:
        logger.info("Getting full historical data...")
        historical_data = data.get_historical_data()
        if historical_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No historical data available",
            )

        hist_data = data.get_historical_data()
        if not hist_data:
            raise RuntimeError("No historical data loaded")
        
        offline_data = hist_data["offline_data"]
        hist_rbp = hist_data["hist_rbp"]
        hist_rr = hist_data["hist_rr"]
        hist_fpr = hist_data["hist_fpr"]
        
        smoothed_rbp = hist_data["smoothed_rbp"]
        smoothed_rr = hist_data["smoothed_rr"]
        smoothed_fpr = hist_data["smoothed_fpr"]
        
        start_date = hist_data["start_date"]
        current_date = hist_data["current_date"]


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


def main():
    """Entry point for running the server."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

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
