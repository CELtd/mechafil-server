Configuration
=============

This page documents all configuration options for the MechaFil Server.

Environment Variables
---------------------

Server Settings
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``HOST``
     - ``0.0.0.0``
     - Server host address
   * - ``PORT``
     - ``8000``
     - Server port number
   * - ``RELOAD``
     - ``false``
     - Enable auto-reload on code changes (development only)
   * - ``LOG_LEVEL``
     - ``INFO``
     - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

API Authentication
~~~~~~~~~~~~~~~~~~

The server requires Spacescope API credentials to fetch historical blockchain data.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``SPACESCOPE_TOKEN``
     - Bearer token string (e.g., ``Bearer YOUR_TOKEN_HERE``)
   * - ``SPACESCOPE_AUTH_FILE``
     - Path to JSON file containing auth key (default: ``.spacescope_auth``)

**Note**: You only need to set ONE of these variables. The server will use ``SPACESCOPE_TOKEN`` if available, otherwise it will look for ``SPACESCOPE_AUTH_FILE``.

**Auth File Format**:

.. code-block:: json

   {
     "auth_key": "Bearer YOUR_TOKEN_HERE"
   }

CORS Settings
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``CORS_ORIGINS``
     - ``*``
     - Comma-separated list of allowed origins

**Example**:

.. code-block:: bash

   CORS_ORIGINS=http://localhost:3000,https://app.example.com

Data Refresh Settings
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``RELOAD_TRIGGER``
     - ``02:00``
     - Daily data refresh time in UTC (HH:MM format)
   * - ``RELOAD_TEST_MODE``
     - ``false``
     - Enable test mode (refreshes every 2 minutes instead of daily)

**Example**:

.. code-block:: bash

   # Refresh data at 3:30 AM UTC
   RELOAD_TRIGGER=03:30

   # Enable test mode for development
   RELOAD_TEST_MODE=true

Configuration File
------------------

Create a ``.env`` file in the repository root or in the ``mechafil-server`` directory:

.. code-block:: bash

   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=INFO

   # Spacescope Authentication
   SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
   # OR
   # SPACESCOPE_AUTH_FILE=./auths/spacescope_auth.json

   # CORS Settings
   CORS_ORIGINS=*

   # Data Refresh
   RELOAD_TRIGGER=02:00
   RELOAD_TEST_MODE=false

The server automatically loads environment variables from:

1. ``.env`` in the repository root
2. ``.env`` in the ``mechafil-server`` directory
3. ``.test-env`` in the repository root

Application Constants
---------------------

These constants are defined in ``mechafil_server/config.py`` and can be modified if needed:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Constant
     - Default
     - Description
   * - ``STARTUP_DATE``
     - ``2025-01-01``
     - Historical data start date
   * - ``WINDOW_DAYS``
     - ``3650``
     - Default forecast window (10 years)
   * - ``SECTOR_DURATION_DAYS``
     - ``540``
     - Default sector duration in days
   * - ``LOCK_TARGET``
     - ``0.3``
     - Default lock target ratio
   * - ``MAX_HISTORICAL_DATA_FETCHING_RETRIES``
     - ``10``
     - Maximum retries when fetching historical data

Cache Configuration
-------------------

The server uses DiskCache for persistent caching of historical data.

Cache Directory
~~~~~~~~~~~~~~~

Default location: ``mechafil-server/.cache``

The cache directory is automatically created on first startup.

Cache Keys
~~~~~~~~~~

Cache keys are generated based on the date range:

.. code-block:: python

   cache_key = f"offline_data_{start_date}{current_date}{end_date}"

Cache Invalidation
~~~~~~~~~~~~~~~~~~

The cache is automatically invalidated and refreshed:

* Daily at the configured ``RELOAD_TRIGGER`` time
* When the server restarts and no valid cache exists for the current date
* After ``MAX_HISTORICAL_DATA_FETCHING_RETRIES`` attempts with different dates

Manual Cache Clearing
~~~~~~~~~~~~~~~~~~~~~

To manually clear the cache:

.. code-block:: bash

   rm -rf mechafil-server/.cache

The server will fetch fresh data on the next request.

Production Configuration
------------------------

Recommended Settings
~~~~~~~~~~~~~~~~~~~~

For production deployments:

.. code-block:: bash

   # Production .env
   HOST=0.0.0.0
   PORT=8000
   RELOAD=false
   LOG_LEVEL=INFO

   # Use specific CORS origins (not *)
   CORS_ORIGINS=https://app.example.com,https://dashboard.example.com

   # Production auth
   SPACESCOPE_AUTH_FILE=/etc/secrets/spacescope_auth.json

   # Daily refresh at off-peak hours
   RELOAD_TRIGGER=03:00

Security Considerations
~~~~~~~~~~~~~~~~~~~~~~~

1. **Never commit credentials** - Use ``.gitignore`` to exclude:

   * ``.env``
   * ``*_auth.json``
   * ``.cache/``

2. **Restrict CORS** - Set specific allowed origins, not ``*``

3. **Use HTTPS** - Deploy behind a reverse proxy (nginx, Caddy) with SSL

4. **Secure auth files** - Set appropriate file permissions:

   .. code-block:: bash

      chmod 600 /etc/secrets/spacescope_auth.json

5. **Environment isolation** - Use environment-specific ``.env`` files

Running with Docker
~~~~~~~~~~~~~~~~~~~

Example Dockerfile configuration:

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   # Install dependencies
   COPY pyproject.toml poetry.lock ./
   RUN pip install poetry && poetry install --no-dev

   # Copy application
   COPY mechafil_server ./mechafil_server

   # Environment variables (override with docker run -e or compose)
   ENV HOST=0.0.0.0
   ENV PORT=8000
   ENV LOG_LEVEL=INFO

   EXPOSE 8000

   CMD ["poetry", "run", "mechafil-server"]

Docker Compose example:

.. code-block:: yaml

   version: '3.8'

   services:
     mechafil-server:
       build: .
       ports:
         - "8000:8000"
       environment:
         - HOST=0.0.0.0
         - PORT=8000
         - LOG_LEVEL=INFO
         - SPACESCOPE_TOKEN=${SPACESCOPE_TOKEN}
         - CORS_ORIGINS=${CORS_ORIGINS}
         - RELOAD_TRIGGER=02:00
       volumes:
         - ./cache:/app/.cache
       restart: unless-stopped

Run with:

.. code-block:: bash

   docker-compose up -d

Development Configuration
-------------------------

Recommended Settings
~~~~~~~~~~~~~~~~~~~~

For local development:

.. code-block:: bash

   # Development .env
   HOST=127.0.0.1
   PORT=8000
   RELOAD=true
   LOG_LEVEL=DEBUG

   # Allow all CORS in development
   CORS_ORIGINS=*

   # Use local auth file
   SPACESCOPE_AUTH_FILE=.spacescope_auth

   # Test mode for faster refresh cycles
   RELOAD_TEST_MODE=true

Hot Reload
~~~~~~~~~~

Enable auto-reload for development:

.. code-block:: bash

   RELOAD=true poetry run uvicorn mechafil_server.main:app --reload

Or use the development script:

.. code-block:: bash

   poetry run uvicorn mechafil_server.main:app --reload --host 127.0.0.1 --port 8000

Logging Configuration
---------------------

Log Levels
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Use Case
   * - ``DEBUG``
     - Development, detailed diagnostic information
   * - ``INFO``
     - Production, general informational messages
   * - ``WARNING``
     - Production, warning messages for potentially harmful situations
   * - ``ERROR``
     - Production, error messages for serious problems
   * - ``CRITICAL``
     - Production, critical messages for very serious errors

Log Format
~~~~~~~~~~

Default format:

.. code-block:: text

   %(asctime)s - %(name)s - %(levelname)s - %(message)s

Example output:

.. code-block:: text

   2025-10-03 12:34:56,789 - mechafil_server.main - INFO - Starting up Mechafil Server...
   2025-10-03 12:34:57,123 - mechafil_server.data - INFO - Historical data loaded successfully

Monitoring and Health Checks
-----------------------------

Health Check Endpoint
~~~~~~~~~~~~~~~~~~~~~

Use ``/health`` for monitoring:

.. code-block:: bash

   curl http://localhost:8000/health

Response includes:

* Service status
* Version information
* JAX backend (CPU/GPU)

Kubernetes Liveness/Readiness Probes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   livenessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 30
     periodSeconds: 10

   readinessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 10
     periodSeconds: 5

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: Server fails to start with "Missing Spacescope auth" error

**Solution**: Ensure ``SPACESCOPE_TOKEN`` or ``SPACESCOPE_AUTH_FILE`` is set correctly

---

**Issue**: Historical data not loading

**Solution**:
1. Check Spacescope credentials
2. Check network connectivity
3. Review logs for specific errors
4. Try clearing cache: ``rm -rf .cache``

---

**Issue**: CORS errors in browser

**Solution**: Add your frontend origin to ``CORS_ORIGINS``

.. code-block:: bash

   CORS_ORIGINS=http://localhost:3000,https://app.example.com

---

**Issue**: Slow simulation performance

**Solution**:
1. Reduce ``forecast_length_days``
2. Use ``output`` parameter to request only needed fields
3. Consider using GPU acceleration (install JAX with CUDA support)

Debug Mode
~~~~~~~~~~

Enable detailed logging:

.. code-block:: bash

   LOG_LEVEL=DEBUG poetry run mechafil-server

This will show:

* Detailed request/response information
* JAX compilation steps
* Cache operations
* Data processing steps

Next Steps
----------

* Learn about :doc:`deployment` options
* Review :doc:`api/endpoints` for API usage
* Check :doc:`examples/quickstart` for getting started
