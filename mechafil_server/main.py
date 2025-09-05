"""Main FastAPI application for the Mechafil Server."""

import json
import base64
from io import BytesIO
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from .models import (
    HealthResponse,
    ErrorResponse,
    SimulationRequest,
    SimulationError,
)
from .simulation import SimulationRunner
from mechafil_jax import sim as mechafil_sim
from mechafil_jax import constants as MF_CONST

# Load environment variables from common locations
load_dotenv()  # default: current working directory
try:
    here = Path(__file__).resolve()
    server_dir = here.parent
    repo_root = server_dir.parent
    # Load from repo root, server dir, and optional test env if present
    for env_path in [
        repo_root / ".env",
        server_dir / ".env",
        repo_root / ".test-env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
except Exception:
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global simulation runner
simulation_runner = None

# 10-year window in days
WINDOW_DAYS = 365 * 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global simulation_runner
    
    # Startup
    logger.info("Starting up Mechafil Server...")
    logger.info(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    logger.info(f"JAX devices: {jax.devices()}")
    
    # Initialize simulation runner
    simulation_runner = SimulationRunner()
    logger.info("Simulation runner initialized")
    
    # Load historical data on startup
    logger.info("Loading historical data on startup...")
    try:
        simulation_runner.load_historical_data()
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
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        jax_backend = jax.lib.xla_bridge.get_backend().platform
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            jax_backend=jax_backend
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
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
            "simulate": "/simulate (POST) - Simulation with default values from template-request.json",
        },
        "template_info": "Empty request '{}' uses defaults from template-request.json",
        "quick_test": "curl -X POST http://localhost:8000/simulate -H 'Content-Type: application/json' -d '{}'",
        "historical_data": "/historical-data (GET) - View loaded historical data summary",
        "historical_data_full": "/historical-data/full (GET) - View all historical data values"
    }


@app.get("/historical-data", tags=["Data"])
async def get_historical_data():
    """Get pretty printed historical data if available."""
    global simulation_runner
    
    if simulation_runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Simulation runner not initialized"
        )
    
    historical_data_file = simulation_runner.data_dir / "historical_data.pkl"
    
    if not historical_data_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Historical data file not found. Server may still be loading data."
        )
    
    try:
        summary = simulation_runner.get_historical_data_summary()
        return {
            "message": "Historical data loaded and available",
            "file_path": str(historical_data_file),
            "file_size_mb": round(historical_data_file.stat().st_size / (1024 * 1024), 2),
            "data_summary": summary
        }
    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical data: {str(e)}"
        )


@app.get("/historical-data/full", tags=["Data"])
async def get_historical_data_full():
    """Get complete historical data with all values."""
    global simulation_runner
    
    if simulation_runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Simulation runner not initialized"
        )
    
    historical_data_file = simulation_runner.data_dir / "historical_data.pkl"
    
    if not historical_data_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Historical data file not found. Server may still be loading data."
        )
    
    try:
        logger.info("Getting historical data from simulation runner...")
        historical_data = simulation_runner.get_historical_data()
        if historical_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No historical data available"
            )
            
        logger.info("Unpacking historical data...")
        offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, hist_rbp, hist_rr, hist_fpr = historical_data
        logger.info("Historical data unpacked successfully")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            import numpy as np  # Import within function to ensure availability
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
                return float(obj) if 'float' in str(type(obj)) else int(obj)
            else:
                return obj
        
        logger.info("Converting data to serializable format...")
        
        # Simple conversion without complex recursion
        offline_data_converted = {}
        for key, value in offline_data.items():
            try:
                if hasattr(value, 'tolist'):
                    offline_data_converted[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    offline_data_converted[key] = [float(x) if hasattr(x, 'tolist') or isinstance(x, (np.number, np.floating, np.integer)) else x for x in value]
                else:
                    offline_data_converted[key] = float(value) if isinstance(value, (np.number, np.floating, np.integer)) else value
            except Exception as conv_error:
                logger.warning(f"Failed to convert {key}: {conv_error}")
                offline_data_converted[key] = str(value)  # fallback to string representation
        
        logger.info("Data conversion completed")
        
        return {
            "message": "Complete historical data",
            "file_path": str(historical_data_file),
            "file_size_mb": round(historical_data_file.stat().st_size / (1024 * 1024), 2),
            "smoothed_metrics": {
                "raw_byte_power": float(smoothed_rbp),
                "renewal_rate": float(smoothed_rr),
                "filplus_rate": float(smoothed_fpr)
            },
            "historical_arrays": {
                "raw_byte_power": [float(x) for x in hist_rbp],
                "renewal_rate": [float(x) for x in hist_rr],
                "filplus_rate": [float(x) for x in hist_fpr]
            },
            "offline_data": offline_data_converted
        }
        
    except Exception as e:
        logger.error(f"Error retrieving full historical data: {e}")
        logger.exception("Full traceback for historical data error:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving historical data: {str(e)}"
        )


def _to_jax_vector(value, length, name):
    """Convert scalar or list to a Python list of length 'length'.

    - float/int: repeat into length
    - list length 1: treat as constant
    - list length == length: use as-is
    - otherwise: raise ValueError
    """
    import numbers
    if value is None:
        return None
    if isinstance(value, numbers.Number):
        return [float(value)] * length
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"{name} cannot be an empty list")
        if len(value) == 1:
            return [float(value[0])] * length
        if len(value) == length:
            return [float(x) for x in value]
        raise ValueError(f"{name} length {len(value)} does not match forecast_length_days {length}")
    raise ValueError(f"Unsupported type for {name}: {type(value)}")


@app.get("/sim-dates", tags=["Debug"])
async def sim_dates():
    """Expose the effective start/current dates and historical lengths for debugging."""
    global simulation_runner
    if simulation_runner is None:
        raise HTTPException(status_code=503, detail="Simulation runner not initialized")
    sd = getattr(simulation_runner, 'start_date', None)
    cd = getattr(simulation_runner, 'current_date', None)
    hd = simulation_runner.historical_data
    H = hist_rr_len = None
    if hd is not None:
        try:
            offline_data = hd[0]
            H = len(offline_data.get("historical_raw_power_eib", []))
            hist_rr_len = len(offline_data.get("historical_renewal_rate", []))
        except Exception:
            pass
    return {
        "start_date": sd.isoformat() if sd else None,
        "current_date": cd.isoformat() if cd else None,
        "historical_raw_power_len": H,
        "historical_renewal_rate_len": hist_rr_len,
    }


@app.post("/simulate", tags=["Simulation"])
async def simulate(req: SimulationRequest):
    """
    Run a Filecoin forecast simulation.

    Example curl:
      curl -X POST http://localhost:8000/simulate \
        -H 'Content-Type: application/json' \
        -d '{"forecast_length_days": 3650, "lock_target": 0.3}'
    """
    global simulation_runner

    if simulation_runner is None or simulation_runner.historical_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical data not loaded yet; try again shortly"
        )

    # Enforce window
    forecast_len = req.forecast_length_days if req.forecast_length_days is not None else WINDOW_DAYS
    if forecast_len > WINDOW_DAYS:
        return SimulationError(
            message=f"forecast_length_days {forecast_len} exceeds WINDOW {WINDOW_DAYS}",
            detail={"WINDOW_DAYS": WINDOW_DAYS}
        )

    # Extract offline data and smoothed metrics from cache
    try:
        offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, *_ = simulation_runner.historical_data
    except Exception as e:
        logger.error(f"Cached historical data unpack failed: {e}")
        raise HTTPException(status_code=500, detail="Corrupted historical data cache")

    # Defaults from tests/repo conventions
    # - lock_target: 0.3
    # - sector_duration_days: 540
    lock_target = req.lock_target if req.lock_target is not None else 0.3
    sector_duration = req.sector_duration_days if req.sector_duration_days is not None else 540

    # Default vectors: use smoothed last-30-days medians when not provided
    # Ensure forecast vectors match the minimum length needed for JAX calculations
    min_forecast_len = max(78, forecast_len)  # Use same buffer as scheduled expiration vectors
    try:
        rbp_vec = _to_jax_vector(req.rbp if req.rbp is not None else smoothed_rbp, min_forecast_len, "rbp")
        rr_vec = _to_jax_vector(req.rr if req.rr is not None else smoothed_rr, min_forecast_len, "rr")
        fpr_vec = _to_jax_vector(req.fpr if req.fpr is not None else smoothed_fpr, min_forecast_len, "fpr")
    except ValueError as ve:
        return SimulationError(message=str(ve))

    # Dates must match the ones used at startup
    if simulation_runner.start_date is None or simulation_runner.current_date is None:
        logger.error("SimulationRunner dates not set")
        raise HTTPException(status_code=500, detail="Server missing startup dates")

    start_date = simulation_runner.start_date
    current_date = simulation_runner.current_date

    # Sanity checks on historical lengths and dates
    H = len(offline_data.get("historical_raw_power_eib", []))
    hist_rr_len = len(offline_data.get("historical_renewal_rate", []))
    if H == 0 or hist_rr_len == 0:
        raise HTTPException(status_code=500, detail="Historical arrays are empty; reload historical data")
    if hist_rr_len != H - 1:
        msg = {
            "message": "Historical length mismatch",
            "historical_raw_power_len": H,
            "historical_renewal_rate_len": hist_rr_len,
            "expected_historical_renewal_rate_len": H - 1,
            "note": "Delete data/historical_data.pkl and restart to refresh cache if mismatch persists."
        }
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    expected_sim_len = (current_date - start_date).days + forecast_len
    if expected_sim_len <= 0:
        raise HTTPException(status_code=500, detail={
            "message": "Invalid simulation length computed",
            "expected_sim_len": expected_sim_len,
            "start_date": start_date.isoformat(),
            "current_date": current_date.isoformat(),
            "forecast_len": forecast_len,
        })

    # Run the simulation
    try:
        # Ensure JAX arrays for functions that call `.cumsum()` as a method
        rbp_arr = jnp.array(rbp_vec)
        rr_arr = jnp.array(rr_vec)
        fpr_arr = jnp.array(fpr_vec)

        # Extract the portion of pre-loaded data needed for this simulation window
        # The data was loaded once for 10 years, now we extract only what's needed for forecast_len
        offline_data_windowed = offline_data.copy()
        
        # Trim scheduled expiration vectors to match the forecast window
        # Add buffer for JAX lookback calculations (minimum 78 days discovered through testing)
        LOOKBACK_BUFFER = max(78, forecast_len)  # Ensure at least 78 days or forecast length, whichever is larger
        effective_length = min(LOOKBACK_BUFFER, len(offline_data_windowed.get('rb_known_scheduled_expire_vec', [])))
        
        if 'rb_known_scheduled_expire_vec' in offline_data_windowed:
            rb_expire_vec = offline_data_windowed['rb_known_scheduled_expire_vec']
            logger.info(f"rb_expire_vec original length: {len(rb_expire_vec)}, forecast_len: {forecast_len}, effective_length: {effective_length}")
            if len(rb_expire_vec) > effective_length:
                offline_data_windowed['rb_known_scheduled_expire_vec'] = rb_expire_vec[:effective_length]
                logger.info(f"rb_expire_vec trimmed to length: {len(offline_data_windowed['rb_known_scheduled_expire_vec'])}")
                
        if 'qa_known_scheduled_expire_vec' in offline_data_windowed:
            qa_expire_vec = offline_data_windowed['qa_known_scheduled_expire_vec']
            logger.info(f"qa_expire_vec original length: {len(qa_expire_vec)}, forecast_len: {forecast_len}, effective_length: {effective_length}")
            if len(qa_expire_vec) > effective_length:
                offline_data_windowed['qa_known_scheduled_expire_vec'] = qa_expire_vec[:effective_length]
                logger.info(f"qa_expire_vec trimmed to length: {len(offline_data_windowed['qa_known_scheduled_expire_vec'])}")
                
        # For pledge release, use forecast_len as it doesn't need the same lookback buffer  
        if 'known_scheduled_pledge_release_full_vec' in offline_data_windowed:
            pledge_release_vec = offline_data_windowed['known_scheduled_pledge_release_full_vec'] 
            logger.info(f"pledge_release_vec original length: {len(pledge_release_vec)}, forecast_len: {forecast_len}")
            if len(pledge_release_vec) > forecast_len:
                offline_data_windowed['known_scheduled_pledge_release_full_vec'] = pledge_release_vec[:forecast_len]
                logger.info(f"pledge_release_vec trimmed to length: {len(offline_data_windowed['known_scheduled_pledge_release_full_vec'])}")
        
        logger.info(f"Simulation parameters: forecast_len={forecast_len}, rbp_len={len(rbp_arr)}, rr_len={len(rr_arr)}, fpr_len={len(fpr_arr)}")
        
        # Debug: Check input dates to simulation
        logger.info(f"Input dates to simulation:")
        logger.info(f"  start_date: {start_date} (type: {type(start_date)})")
        logger.info(f"  current_date: {current_date} (type: {type(current_date)})")
        logger.info(f"  forecast_len: {forecast_len}")
        logger.info(f"  sector_duration: {sector_duration}")
        logger.info(f"  Historical data length: {len(offline_data_windowed.get('historical_raw_power_eib', []))}")
        logger.info(f"  Days between start and current: {(current_date - start_date).days}")
        logger.info(f"  Expected total simulation days: {(current_date - start_date).days + forecast_len}")

        results = mechafil_sim.run_sim(
            rbp_arr, rr_arr, fpr_arr,
            lock_target,
            start_date,
            current_date,
            min_forecast_len,
            sector_duration,
            offline_data_windowed,
            use_available_supply=False,
        )
        
        # Trim results back to the requested forecast length if we used a buffer
        if min_forecast_len > forecast_len:
            logger.info(f"Trimming results from {min_forecast_len} to {forecast_len} days")
            historical_len = (current_date - start_date).days
            target_total_len = historical_len + forecast_len
            
            for key, value in results.items():
                if hasattr(value, '__len__') and len(value) > target_total_len:
                    results[key] = value[:target_total_len]
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

    # Convert JAX/NumPy arrays to JSON-serializable and scrub NaN/Inf
    def serialize(value):
        import numpy as np
        try:
            if hasattr(value, 'tolist'):
                return value.tolist()
            if isinstance(value, (np.generic,)):
                return value.item()
            return value
        except Exception:
            return str(value)

    def sanitize(obj):
        import math
        import numpy as np
        # Scalars
        if isinstance(obj, (int,)):
            return obj
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        # NumPy scalar
        if 'numpy' in type(obj).__module__:
            try:
                val = float(obj)
                return val if math.isfinite(val) else None
            except Exception:
                return None
        # List/Tuple
        if isinstance(obj, (list, tuple)):
            return [sanitize(x) for x in obj]
        # Dict
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    serialized = {k: serialize(v) for k, v in results.items()}
    serialized = sanitize(serialized)

    # Save setup + results to a file for later retrieval
    end_date = current_date + timedelta(days=forecast_len)

    def convert_offline(obj):
        import numpy as np
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_offline(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_offline(x) for x in obj]
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return obj

    setup_payload = {
        "start_date": start_date.isoformat(),
        "current_date": current_date.isoformat(),
        "end_date": end_date.isoformat(),
        "forecast_length_days": forecast_len,
        "sector_duration_days": sector_duration,
        "lock_target": lock_target,
        "rbp": rbp_arr.tolist(),
        "rr": rr_arr.tolist(),
        "fpr": fpr_arr.tolist(),
        "offline_data": convert_offline(offline_data),
    }

    latest_file = simulation_runner.data_dir / "latest_simulation.json"
    try:
        with open(latest_file, 'w') as f:
            json.dump({
                "setup": setup_payload,
                "results": serialized,
            }, f)
    except Exception as e:
        logger.warning(f"Failed to save latest_simulation.json: {e}")

    return serialized


@app.get("/latest-simulation", tags=["Simulation"])
async def get_latest_simulation():
    """Return the last saved simulation setup and results as JSON."""
    global simulation_runner
    if simulation_runner is None:
        raise HTTPException(status_code=503, detail="Simulation runner not initialized")
    latest_file = simulation_runner.data_dir / "latest_simulation.json"
    if not latest_file.exists():
        raise HTTPException(status_code=404, detail="No saved simulation found. Run POST /simulate first.")
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        # Sanitize NaN/Inf before returning (Starlette disallows them)
        def _sanitize(obj):
            import math
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(x) for x in obj]
            return obj
        sanitized = _sanitize(data)
        # Optionally rewrite file with sanitized values to prevent future errors
        try:
            with open(latest_file, 'w') as f:
                json.dump(sanitized, f)
        except Exception:
            pass
        return sanitized
    except Exception as e:
        logger.error(f"Failed to read latest_simulation.json: {e}")
        raise HTTPException(status_code=500, detail="Failed to read saved simulation file")


@app.get("/latest-simulation/plots", tags=["Simulation"])
async def get_latest_simulation_plots(max_plots: int = 24):
    """Generate plots for each numeric result series from the last simulation.

    Returns JSON with a list of base64-encoded PNGs and metadata.
    """
    global simulation_runner
    if simulation_runner is None:
        raise HTTPException(status_code=503, detail="Simulation runner not initialized")

    latest_file = simulation_runner.data_dir / "latest_simulation.json"
    if not latest_file.exists():
        raise HTTPException(status_code=404, detail="No saved simulation found. Run POST /simulate first.")

    try:
        with open(latest_file, 'r') as f:
            payload = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read latest_simulation.json: {e}")
        raise HTTPException(status_code=500, detail="Failed to read saved simulation file")

    setup = payload.get("setup", {})
    results = payload.get("results", {})

    # Prepare x-axis as day indices and derive dates
    from datetime import date, timedelta
    try:
        start_date = date.fromisoformat(setup.get("start_date"))
        end_date = date.fromisoformat(setup.get("end_date"))
    except Exception:
        start_date = None
        end_date = None

    # Lazy import matplotlib with headless backend
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matplotlib not available: {e}")

    def is_numeric_list(v):
        if not isinstance(v, list) or len(v) < 2:
            return False
        try:
            for x in v[:5]:
                float(x)
            return True
        except Exception:
            return False

    plots = []
    count = 0
    # Ensure stable ordering for reproducibility
    for key in sorted(results.keys()):
        if count >= max_plots:
            break
        series = results[key]
        if not is_numeric_list(series):
            continue

        n = len(series)
        x = list(range(n))
        # Build a handful of date ticks (start, mid, end)
        date_ticks = []
        if start_date is not None:
            try:
                mid = n // 2
                d0 = start_date
                d1 = start_date + timedelta(days=mid)
                d2 = start_date + timedelta(days=n - 1)
                date_ticks = [d0.isoformat(), d1.isoformat(), d2.isoformat()]
            except Exception:
                date_ticks = []

        # Generate plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, series, linewidth=1.2)
        ax.set_title(key)
        ax.set_xlabel("Day Index" + (f"  (Dates: {', '.join(date_ticks)})" if date_ticks else ""))
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)

        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        plots.append({
            "key": key,
            "title": key,
            "xlabel": "Day Index",
            "ylabel": key,
            "length": n,
            "start_date": setup.get("start_date"),
            "end_date": setup.get("end_date"),
            "image_base64": f"data:image/png;base64,{b64}",
        })
        count += 1

    if not plots:
        raise HTTPException(status_code=400, detail="No plottable numeric series found in results")

    return {"plots": plots, "total": len(plots)}


@app.get("/latest-simulation/plots/html", tags=["Simulation"], response_class=HTMLResponse)
async def get_latest_simulation_plots_html(max_plots: int = 24):
    """Render an HTML gallery for the latest simulation plots.

    This is a convenience view that embeds base64 PNGs in a simple grid.
    """
    global simulation_runner
    if simulation_runner is None:
        raise HTTPException(status_code=503, detail="Simulation runner not initialized")

    latest_file = simulation_runner.data_dir / "latest_simulation.json"
    if not latest_file.exists():
        raise HTTPException(status_code=404, detail="No saved simulation found. Run POST /simulate first.")

    try:
        with open(latest_file, 'r') as f:
            payload = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read latest_simulation.json: {e}")
        raise HTTPException(status_code=500, detail="Failed to read saved simulation file")

    setup = payload.get("setup", {})
    results = payload.get("results", {})

    # Prepare x-axis and dates
    from datetime import date, timedelta
    # For plotting, enforce base start date = 2020-10-15
    base_start_date = date(2020, 10, 15)
    try:
        _start_date_saved = date.fromisoformat(setup.get("start_date"))
    except Exception:
        _start_date_saved = None
    try:
        end_date = date.fromisoformat(setup.get("end_date"))
    except Exception:
        end_date = None

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"matplotlib not available: {e}")

    def is_numeric_list(v):
        if not isinstance(v, list) or len(v) < 2:
            return False
        try:
            for x in v[:5]:
                float(x)
            return True
        except Exception:
            return False

    # Generate plots
    items = []
    count = 0
    for key in sorted(results.keys()):
        if count >= max_plots:
            break
        series = results[key]
        if not is_numeric_list(series):
            continue
        n = len(series)
        # Build date x-axis
        fig, ax = plt.subplots(figsize=(6, 3))
        # Some series are forecast-only in the simulator outputs
        forecast_only_keys = {
            "rb_sched_expire_power_pib",
            "qa_sched_expire_power_pib",
        }
        # Try to read current_date once here
        current_date_parsed = None
        if setup.get("current_date"):
            try:
                current_date_parsed = date.fromisoformat(setup.get("current_date"))
            except Exception:
                current_date_parsed = None
        if key in forecast_only_keys and current_date_parsed is not None:
            # For forecast-only keys, dates start at current_date
            dates = [current_date_parsed + timedelta(days=i) for i in range(n)]
        else:
            # Default: dates start at fixed base start date
            dates = [base_start_date + timedelta(days=i) for i in range(n)]

        # Split at current_date: historical (blue), forecast (red)
        # Determine historical length boundary for coloring
        hist_len_offline = None
        try:
            offline_data = setup.get("offline_data") or {}
            if isinstance(offline_data, dict) and "historical_raw_power_eib" in offline_data:
                hist_len_offline = len(offline_data["historical_raw_power_eib"])  # inclusive days from 2020-10-15
        except Exception:
            hist_len_offline = None

        hist_len_main = None
        if setup.get("current_date"):
            try:
                current_date = date.fromisoformat(setup.get("current_date"))
                hist_len_main = (current_date - base_start_date).days + 1
            except Exception:
                hist_len_main = None

        if key in forecast_only_keys:
            hist_len = 0
        else:
            hist_len = hist_len_offline if hist_len_offline is not None else hist_len_main

        if hist_len is None:
            # Unknown split; show in blue
            ax.plot(dates, series, linewidth=1.2, color='tab:blue')
        else:
            # Clamp boundary and color segments
            hist_len = max(0, min(hist_len, n))
            if hist_len == 0:
                ax.plot(dates, series, linewidth=1.2, color='tab:red', label='forecast')
            elif hist_len >= n:
                ax.plot(dates, series, linewidth=1.2, color='tab:blue', label='historical')
            else:
                ax.plot(dates[:hist_len], series[:hist_len], linewidth=1.2, color='tab:blue', label='historical')
                ax.plot(dates[hist_len:], series[hist_len:], linewidth=1.2, color='tab:red', label='forecast')

        # Ticks: major = yearly, minor = monthly
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        # Minor ticks have no labels by default; ensure they show as ticks
        ax.tick_params(axis='x', which='major', labelrotation=0)
        ax.tick_params(axis='x', which='minor', length=3, labelbottom=False)

        # Fix x-limits to a common frame: from base start to setup end_date (if available)
        try:
            if end_date is not None:
                ax.set_xlim([base_start_date, end_date])
        except Exception:
            pass

        # Ensure first, current, and last dates are readable as context in the title
        date_label = f"{dates[0].isoformat()} — {setup.get('current_date')} — {dates[-1].isoformat()}"

        ax.set_title(key)
        ax.set_xlabel("Date (major: yearly, minor: monthly)")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        # Add legend only if we drew two segments
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, frameon=False)
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        items.append((key, date_label, b64))
        count += 1

    if not items:
        raise HTTPException(status_code=400, detail="No plottable numeric series found in results")

    # Compose HTML
    title = "Latest Simulation Plots"
    sd = setup.get("start_date")
    cd = setup.get("current_date")
    ed = setup.get("end_date")
    header = f"<h1>{title}</h1><p>Start: {sd} — Current: {cd} — End: {ed}</p>"
    style = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
      figure { margin: 0; padding: 8px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
      figcaption { font-size: 13px; color: #555; margin-top: 6px; }
      img { max-width: 100%; height: auto; display: block; }
    </style>
    """
    grid_items = []
    for key, date_label, b64 in items:
        cap = f"<strong>{key}</strong>" + (f"<br/><span>{date_label}</span>" if date_label else "")
        grid_items.append(f"<figure><img src='data:image/png;base64,{b64}' alt='{key}'/><figcaption>{cap}</figcaption></figure>")
    grid = "<div class='grid'>" + "\n".join(grid_items) + "</div>"

    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>{style}</head><body>{header}{grid}</body></html>"
    return HTMLResponse(content=html)


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
        log_level="info"
    )


if __name__ == "__main__":
    main()
