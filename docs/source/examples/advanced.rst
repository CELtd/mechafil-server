Advanced Usage
==============

This guide covers advanced usage patterns and techniques for the MechaFil Server API.

Time-Varying Parameters
-----------------------

The simulation engine supports time-varying parameters by passing arrays instead of scalar values. This allows you to model changing network conditions over time.

Array-Based Parameters
~~~~~~~~~~~~~~~~~~~~~~

When you pass an array for ``rbp``, ``rr``, or ``fpr``, the simulation uses those values day-by-day:

.. code-block:: python

   import requests

   # Define parameters that change over time
   params = {
       "forecast_length_days": 5,
       "rbp": [3.0, 3.5, 4.0, 3.8, 3.6],  # Varying onboarding rate
       "rr": [0.8, 0.82, 0.85, 0.85, 0.85],  # Increasing renewal rate
       "fpr": [0.9, 0.9, 0.88, 0.85, 0.85],  # Decreasing FIL+ rate
       "lock_target": 0.3
   }

   response = requests.post("http://localhost:8000/simulate", json=params)
   results = response.json()

Modeling Scenarios
~~~~~~~~~~~~~~~~~~

**Gradual Network Growth**

Model a scenario where onboarding gradually increases:

.. code-block:: python

   import numpy as np

   # Create gradual increase from 3 to 5 PIB/day over 365 days
   days = 365
   rbp_gradual = np.linspace(3.0, 5.0, days).tolist()

   params = {
       "forecast_length_days": days,
       "rbp": rbp_gradual,
       "output": ["network_RBP_EIB", "available_supply"]
   }

   response = requests.post("http://localhost:8000/simulate", json=params)

**Seasonal Patterns**

Model seasonal variations in network activity:

.. code-block:: python

   import numpy as np

   days = 365
   # Create sinusoidal pattern: base 3.5, amplitude 0.5
   seasonal_rbp = (3.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, days))).tolist()

   params = {
       "forecast_length_days": days,
       "rbp": seasonal_rbp,
       "output": ["network_RBP_EIB", "day_network_reward"]
   }

**Market Shock Scenario**

Model sudden changes in network conditions:

.. code-block:: python

   # Normal conditions for 180 days, then shock
   rbp_normal = [3.5] * 180
   rbp_shock = [1.5] * 185  # 50% drop

   params = {
       "forecast_length_days": 365,
       "rbp": rbp_normal + rbp_shock,
       "rr": [0.85] * 180 + [0.6] * 185,  # Lower renewals during shock
       "output": ["available_supply", "network_locked", "day_network_reward"]
   }

Batch Processing
----------------

Run Multiple Simulations
~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple scenarios efficiently:

.. code-block:: python

   import requests
   import concurrent.futures

   BASE_URL = "http://localhost:8000"

   def run_scenario(scenario):
       response = requests.post(f"{BASE_URL}/simulate", json=scenario)
       return scenario['name'], response.json()

   # Define scenarios
   scenarios = [
       {"name": "conservative", "rbp": 2.5, "rr": 0.75, "fpr": 0.8, "forecast_length_days": 365},
       {"name": "baseline", "rbp": 3.5, "rr": 0.85, "fpr": 0.85, "forecast_length_days": 365},
       {"name": "optimistic", "rbp": 5.0, "rr": 0.9, "fpr": 0.9, "forecast_length_days": 365},
   ]

   # Run in parallel
   with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
       results = dict(executor.map(run_scenario, scenarios))

   # Compare final values
   for name, result in results.items():
       final_supply = result['simulation_output']['available_supply'][-1]
       print(f"{name}: Final supply = {final_supply} FIL")

Parameter Sweeps
~~~~~~~~~~~~~~~~

Sweep across parameter ranges to find optimal values:

.. code-block:: python

   import numpy as np
   import requests

   # Sweep lock target from 0.2 to 0.4
   lock_targets = np.linspace(0.2, 0.4, 21)

   results = []
   for target in lock_targets:
       response = requests.post("http://localhost:8000/simulate", json={
           "forecast_length_days": 365,
           "lock_target": float(target),
           "output": ["available_supply", "network_locked", "1y_sector_roi"]
       })

       if response.status_code == 200:
           data = response.json()
           results.append({
               'lock_target': target,
               'final_supply': data['simulation_output']['available_supply'][-1],
               'final_locked': data['simulation_output']['network_locked'][-1],
               'avg_roi': np.mean(data['simulation_output']['1y_sector_roi'])
           })

   # Find optimal lock target (example: maximize available supply)
   optimal = max(results, key=lambda x: x['final_supply'])
   print(f"Optimal lock target: {optimal['lock_target']:.2f}")

Data Visualization
------------------

Plotting with Matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import requests
   import matplotlib.pyplot as plt
   from datetime import datetime, timedelta

   # Run simulation
   response = requests.post("http://localhost:8000/simulate", json={
       "forecast_length_days": 365,
       "output": ["available_supply", "network_RBP_EIB", "day_network_reward"]
   })

   results = response.json()
   output = results['simulation_output']

   # Create date labels (Mondays only, since data is downsampled)
   current_date = datetime.strptime(results['input']['current date'], '%Y-%m-%d')
   # Find next Monday
   days_ahead = (7 - current_date.weekday()) % 7
   first_monday = current_date + timedelta(days=days_ahead)

   num_points = len(output['available_supply'])
   dates = [first_monday + timedelta(weeks=i) for i in range(num_points)]

   # Create figure with subplots
   fig, axes = plt.subplots(3, 1, figsize=(12, 10))

   # Plot available supply
   axes[0].plot(dates, output['available_supply'])
   axes[0].set_title('Available Supply')
   axes[0].set_ylabel('FIL')
   axes[0].grid(True)

   # Plot network power
   axes[1].plot(dates, output['network_RBP_EIB'])
   axes[1].set_title('Network Raw Byte Power')
   axes[1].set_ylabel('EIB')
   axes[1].grid(True)

   # Plot daily rewards
   axes[2].plot(dates, output['day_network_reward'])
   axes[2].set_title('Daily Network Rewards')
   axes[2].set_ylabel('FIL/day')
   axes[2].set_xlabel('Date')
   axes[2].grid(True)

   plt.tight_layout()
   plt.savefig('simulation_results.png')
   plt.show()

Comparing Scenarios
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import requests
   import matplotlib.pyplot as plt

   scenarios = {
       'Low Growth': {'rbp': 2.0, 'rr': 0.75},
       'Medium Growth': {'rbp': 3.5, 'rr': 0.85},
       'High Growth': {'rbp': 5.0, 'rr': 0.9}
   }

   plt.figure(figsize=(12, 6))

   for name, params in scenarios.items():
       response = requests.post("http://localhost:8000/simulate", json={
           **params,
           "forecast_length_days": 365,
           "output": ["available_supply"]
       })

       data = response.json()
       supply = data['simulation_output']['available_supply']
       weeks = list(range(len(supply)))

       plt.plot(weeks, supply, label=name, linewidth=2)

   plt.xlabel('Weeks')
   plt.ylabel('Available Supply (FIL)')
   plt.title('Available Supply Projection - Scenario Comparison')
   plt.legend()
   plt.grid(True)
   plt.savefig('scenario_comparison.png')
   plt.show()

Integration Patterns
--------------------

Automated Daily Reports
~~~~~~~~~~~~~~~~~~~~~~~

Generate daily reports using the latest historical data:

.. code-block:: python

   import requests
   from datetime import datetime
   import json

   def generate_daily_report():
       BASE_URL = "http://localhost:8000"

       # Get latest historical metrics
       hist_response = requests.get(f"{BASE_URL}/historical-data")
       hist_data = hist_response.json()['data']

       # Run 30-day forecast
       sim_response = requests.post(f"{BASE_URL}/simulate", json={
           "forecast_length_days": 30,
           "output": ["available_supply", "network_RBP_EIB", "day_network_reward"]
       })
       sim_results = sim_response.json()

       # Generate report
       report = {
           "date": datetime.now().isoformat(),
           "current_metrics": {
               "rbp": hist_data['raw_byte_power_averaged_over_previous_30days'],
               "rr": hist_data['renewal_rate_averaged_over_previous_30days'],
               "fpr": hist_data['filplus_rate_averaged_over_previous_30days']
           },
           "30_day_projection": {
               "supply_change": (
                   sim_results['simulation_output']['available_supply'][-1] -
                   sim_results['simulation_output']['available_supply'][0]
               ),
               "avg_daily_reward": sum(sim_results['simulation_output']['day_network_reward']) / 30,
               "final_power": sim_results['simulation_output']['network_RBP_EIB'][-1]
           }
       }

       # Save report
       with open(f"report_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
           json.dump(report, f, indent=2)

       return report

   # Run daily (e.g., via cron)
   report = generate_daily_report()
   print(json.dumps(report, indent=2))

Webhook Integration
~~~~~~~~~~~~~~~~~~~

Trigger simulations from external events:

.. code-block:: python

   from flask import Flask, request, jsonify
   import requests

   app = Flask(__name__)
   MECHAFIL_URL = "http://localhost:8000"

   @app.route('/webhook/simulation', methods=['POST'])
   def handle_simulation_webhook():
       # Receive parameters from external system
       webhook_data = request.json

       # Map to simulation parameters
       sim_params = {
           "forecast_length_days": webhook_data.get('days', 365),
           "rbp": webhook_data.get('onboarding_rate'),
           "rr": webhook_data.get('renewal_rate'),
           "fpr": webhook_data.get('filplus_rate'),
           "output": webhook_data.get('metrics', [])
       }

       # Run simulation
       response = requests.post(f"{MECHAFIL_URL}/simulate", json=sim_params)

       if response.status_code == 200:
           results = response.json()
           # Process and return results
           return jsonify({
               "status": "success",
               "results": results
           })
       else:
           return jsonify({
               "status": "error",
               "message": response.text
           }), response.status_code

   if __name__ == '__main__':
       app.run(port=5000)

Caching and Performance
-----------------------

Client-Side Caching
~~~~~~~~~~~~~~~~~~~

Implement caching to reduce API calls:

.. code-block:: python

   import requests
   import hashlib
   import json
   from functools import lru_cache

   class MechafilClient:
       def __init__(self, base_url="http://localhost:8000"):
           self.base_url = base_url

       @lru_cache(maxsize=128)
       def get_historical_data(self):
           """Cached historical data fetch"""
           response = requests.get(f"{self.base_url}/historical-data")
           return response.json()

       def simulate_cached(self, **params):
           """Cache simulation results by parameter hash"""
           # Create cache key from parameters
           cache_key = hashlib.md5(
               json.dumps(params, sort_keys=True).encode()
           ).hexdigest()

           # Check cache file
           cache_file = f".cache/{cache_key}.json"
           try:
               with open(cache_file, 'r') as f:
                   return json.load(f)
           except FileNotFoundError:
               pass

           # Fetch from API
           response = requests.post(f"{self.base_url}/simulate", json=params)
           results = response.json()

           # Save to cache
           import os
           os.makedirs('.cache', exist_ok=True)
           with open(cache_file, 'w') as f:
               json.dump(results, f)

           return results

   # Usage
   client = MechafilClient()
   hist_data = client.get_historical_data()  # Cached in memory
   sim_results = client.simulate_cached(
       forecast_length_days=365,
       output=["available_supply"]
   )  # Cached to disk

Error Recovery
--------------

Retry Logic
~~~~~~~~~~~

Implement robust retry mechanisms:

.. code-block:: python

   import requests
   from requests.adapters import HTTPAdapter
   from urllib3.util.retry import Retry

   def create_session_with_retries():
       session = requests.Session()

       # Configure retry strategy
       retry = Retry(
           total=5,
           backoff_factor=1,
           status_forcelist=[500, 502, 503, 504],
           allowed_methods=["GET", "POST"]
       )

       adapter = HTTPAdapter(max_retries=retry)
       session.mount('http://', adapter)
       session.mount('https://', adapter)

       return session

   # Usage
   session = create_session_with_retries()
   response = session.post(
       "http://localhost:8000/simulate",
       json={"forecast_length_days": 365},
       timeout=30
   )

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

Handle service unavailability:

.. code-block:: python

   import requests
   from datetime import datetime

   def get_simulation_with_fallback(params, fallback_data=None):
       try:
           response = requests.post(
               "http://localhost:8000/simulate",
               json=params,
               timeout=10
           )
           response.raise_for_status()
           return response.json()

       except requests.exceptions.RequestException as e:
           print(f"Simulation failed: {e}")

           if fallback_data:
               print("Using fallback data")
               return fallback_data

           # Return minimal valid response
           return {
               "error": True,
               "message": str(e),
               "timestamp": datetime.now().isoformat()
           }

   # Usage with fallback
   results = get_simulation_with_fallback(
       {"forecast_length_days": 365},
       fallback_data={"simulation_output": {"available_supply": []}}
   )

Best Practices
--------------

1. **Always specify output fields** when you don't need all metrics - this reduces response size and improves performance

2. **Use appropriate forecast lengths** - longer forecasts take more time to compute

3. **Implement proper error handling** - network issues and validation errors can occur

4. **Cache historical data** - it only updates daily, so cache it on the client side

5. **Use batch processing** for multiple scenarios - run simulations in parallel when possible

6. **Monitor API health** - regularly check the ``/health`` endpoint

7. **Validate parameters** before sending - ensure arrays have correct lengths and values are in valid ranges

Next Steps
----------

* Review the :doc:`../api/endpoints` for complete API reference
* Check :doc:`../api/models` for detailed data structures
* See :doc:`../configuration` for server setup options
* Learn about :doc:`../deployment` for production usage
