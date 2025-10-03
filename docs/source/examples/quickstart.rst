Quick Start Guide
=================

This guide will help you get started with the MechaFil Server API.

Basic Usage
-----------

Health Check
~~~~~~~~~~~~

First, verify the server is running:

.. code-block:: bash

   curl http://localhost:8000/health

Response:

.. code-block:: json

   {
     "status": "healthy",
     "version": "0.1.0",
     "jax_backend": "cpu"
   }

Get Historical Data
~~~~~~~~~~~~~~~~~~~

Retrieve processed historical Filecoin network data:

.. code-block:: bash

   curl http://localhost:8000/historical-data

This returns downsampled historical metrics including:

* 30-day averaged values for RBP, renewal rate, and FIL+ rate
* Time series arrays for historical network data
* Offline model initialization data

Sample response:

.. code-block:: json

   {
     "data": {
       "raw_byte_power_averaged_over_previous_30days": 3.379532,
       "renewal_rate_averaged_over_previous_30days": 0.834245,
       "filplus_rate_averaged_over_previous_30days": 0.855880,
       "raw_byte_power": [3.2, 3.3, 3.4, ...],
       "renewal_rate": [0.81, 0.82, 0.83, ...],
       "filplus_rate": [0.84, 0.85, 0.86, ...]
     }
   }

Running Simulations
-------------------

Default Simulation
~~~~~~~~~~~~~~~~~~

Run a simulation with all default parameters:

.. code-block:: bash

   curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{}'

This uses:

* Default forecast length: 3650 days (10 years)
* Smoothed historical values for RBP, renewal rate, and FIL+ rate
* Default lock target: 0.3
* Default sector duration: 540 days

Custom Parameters
~~~~~~~~~~~~~~~~~

Run a 1-year forecast with custom parameters:

.. code-block:: bash

   curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{
       "forecast_length_days": 365,
       "rbp": 3.5,
       "rr": 0.85,
       "fpr": 0.9,
       "lock_target": 0.25
     }'

Filter Specific Fields
~~~~~~~~~~~~~~~~~~~~~~

Request only specific output fields to reduce response size:

.. code-block:: bash

   # Single field
   curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{
       "forecast_length_days": 365,
       "output": "available_supply"
     }'

   # Multiple fields
   curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{
       "forecast_length_days": 365,
       "output": ["available_supply", "network_RBP_EIB", "circ_supply"]
     }'

Python Examples
---------------

Using Requests
~~~~~~~~~~~~~~

.. code-block:: python

   import requests

   # Base URL
   BASE_URL = "http://localhost:8000"

   # Health check
   response = requests.get(f"{BASE_URL}/health")
   print(response.json())

   # Get historical data
   response = requests.get(f"{BASE_URL}/historical-data")
   historical_data = response.json()
   print(f"30-day avg RBP: {historical_data['data']['raw_byte_power_averaged_over_previous_30days']}")

   # Run simulation with defaults
   response = requests.post(f"{BASE_URL}/simulate", json={})
   results = response.json()
   print(f"Forecast length: {results['input']['forecast_length_days']} days")

   # Run simulation with custom parameters
   params = {
       "forecast_length_days": 365,
       "rbp": 3.5,
       "rr": 0.85,
       "fpr": 0.9,
       "lock_target": 0.3,
       "output": ["available_supply", "network_RBP_EIB"]
   }
   response = requests.post(f"{BASE_URL}/simulate", json=params)
   results = response.json()

   # Extract results
   available_supply = results['simulation_output']['available_supply']
   network_power = results['simulation_output']['network_RBP_EIB']
   print(f"Final available supply: {available_supply[-1]} FIL")

Using HTTPX (Async)
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import httpx
   import asyncio

   async def run_simulation():
       async with httpx.AsyncClient() as client:
           # Run simulation
           response = await client.post(
               "http://localhost:8000/simulate",
               json={
                   "forecast_length_days": 365,
                   "output": ["available_supply", "circ_supply"]
               }
           )
           results = response.json()
           return results

   # Run async
   results = asyncio.run(run_simulation())
   print(results['simulation_output'])

JavaScript Examples
-------------------

Using Fetch API
~~~~~~~~~~~~~~~

.. code-block:: javascript

   // Health check
   fetch('http://localhost:8000/health')
     .then(response => response.json())
     .then(data => console.log(data));

   // Get historical data
   fetch('http://localhost:8000/historical-data')
     .then(response => response.json())
     .then(data => {
       console.log('30-day avg RBP:',
         data.data.raw_byte_power_averaged_over_previous_30days);
     });

   // Run simulation
   fetch('http://localhost:8000/simulate', {
     method: 'POST',
     headers: {
       'Content-Type': 'application/json'
     },
     body: JSON.stringify({
       forecast_length_days: 365,
       rbp: 3.5,
       rr: 0.85,
       fpr: 0.9,
       output: ['available_supply', 'network_RBP_EIB']
     })
   })
   .then(response => response.json())
   .then(results => {
     console.log('Input params:', results.input);
     console.log('Available supply:', results.simulation_output.available_supply);
   });

Using Axios
~~~~~~~~~~~

.. code-block:: javascript

   const axios = require('axios');

   const BASE_URL = 'http://localhost:8000';

   // Run simulation
   async function runSimulation() {
     try {
       const response = await axios.post(`${BASE_URL}/simulate`, {
         forecast_length_days: 365,
         rbp: 3.5,
         rr: 0.85,
         fpr: 0.9,
         output: ['available_supply', 'network_RBP_EIB']
       });

       const { input, simulation_output } = response.data;
       console.log('Simulation input:', input);
       console.log('Available supply:', simulation_output.available_supply);
     } catch (error) {
       console.error('Error:', error.response.data);
     }
   }

   runSimulation();

Common Use Cases
----------------

Scenario 1: Current Network Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the latest network metrics and run a short-term forecast:

.. code-block:: bash

   # Get current metrics
   curl http://localhost:8000/historical-data | jq '.data | {
     rbp: .raw_byte_power_averaged_over_previous_30days,
     rr: .renewal_rate_averaged_over_previous_30days,
     fpr: .filplus_rate_averaged_over_previous_30days
   }'

   # Run 90-day forecast with current metrics
   curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{
       "forecast_length_days": 90,
       "output": ["available_supply", "network_locked", "day_network_reward"]
     }'

Scenario 2: What-If Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare different lock target scenarios:

.. code-block:: python

   import requests

   BASE_URL = "http://localhost:8000"

   # Test different lock targets
   lock_targets = [0.2, 0.3, 0.4]
   results = {}

   for target in lock_targets:
       response = requests.post(f"{BASE_URL}/simulate", json={
           "forecast_length_days": 365,
           "lock_target": target,
           "output": ["available_supply", "network_locked"]
       })
       results[target] = response.json()['simulation_output']

   # Compare results
   for target, output in results.items():
       final_supply = output['available_supply'][-1]
       final_locked = output['network_locked'][-1]
       print(f"Lock target {target}: Supply={final_supply}, Locked={final_locked}")

Scenario 3: Long-term Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run a 10-year forecast to analyze long-term trends:

.. code-block:: bash

   curl -X POST http://localhost:8000/simulate \
     -H 'Content-Type: application/json' \
     -d '{
       "forecast_length_days": 3650,
       "output": [
         "available_supply",
         "circ_supply",
         "network_RBP_EIB",
         "day_network_reward",
         "1y_sector_roi"
       ]
     }'

Error Handling
--------------

Always check for errors in production code:

.. code-block:: python

   import requests

   def run_safe_simulation(params):
       try:
           response = requests.post(
               "http://localhost:8000/simulate",
               json=params,
               timeout=30
           )
           response.raise_for_status()
           return response.json()
       except requests.exceptions.HTTPError as e:
           print(f"HTTP error: {e.response.status_code}")
           print(f"Details: {e.response.json()}")
       except requests.exceptions.Timeout:
           print("Request timed out")
       except requests.exceptions.RequestException as e:
           print(f"Request failed: {e}")

   # Use it
   results = run_safe_simulation({
       "forecast_length_days": 365,
       "output": "invalid_field"  # This will cause a validation error
   })

Next Steps
----------

* Learn about :doc:`advanced` usage patterns
* Explore the full :doc:`../api/endpoints` reference
* Check :doc:`../api/models` for detailed data structures
