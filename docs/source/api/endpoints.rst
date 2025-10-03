API Endpoints
=============

This page documents all available REST API endpoints provided by the MechaFil Server.

Base URL
--------

When running locally: ``http://localhost:8000``

Root Endpoint
-------------

GET /
~~~~~

Returns basic server information and available endpoints.

**Response**

.. code-block:: json

   {
     "message": "Mechafil Server",
     "version": "0.1.0",
     "docs": "/docs",
     "redoc": "/redoc",
     "endpoints": {
       "health": "/health (GET) - Server health check and JAX backend info",
       "historical_data": "/historical-data (GET) - Historical data downsampled every week",
       "simulate": "/simulate (POST) - Run Filecoin forecast simulation downsampled every week"
     },
     "quick_test": "curl -X POST http://localhost:8000/simulate -H 'Content-Type: application/json' -d '{}'",
     "template_info": "Empty request '{}' uses defaults from historical data"
   }

Health Check
------------

GET /health
~~~~~~~~~~~

Health check endpoint that returns server status and JAX backend information.

**Response Model**: ``HealthResponse``

**Success Response** (200 OK)

.. code-block:: json

   {
     "status": "healthy",
     "version": "0.1.0",
     "jax_backend": "cpu"
   }

**Error Response** (503 Service Unavailable)

Service is unhealthy or experiencing issues.

Historical Data
---------------

GET /historical-data
~~~~~~~~~~~~~~~~~~~~

Retrieves historical Filecoin network data downsampled to Mondays for efficient visualization.

**Response Structure**

The response contains a ``data`` object with three categories of fields:

1. **30-day Averaged Metrics** (scalars)

   * ``raw_byte_power_averaged_over_previous_30days`` - Median RBP over last 30 days (float)
   * ``renewal_rate_averaged_over_previous_30days`` - Median renewal rate over last 30 days (float)
   * ``filplus_rate_averaged_over_previous_30days`` - Median FIL+ rate over last 30 days (float)

2. **Daily Historical Arrays** (lists of floats, downsampled to Mondays)

   * ``raw_byte_power`` - Raw byte power onboarding (PIB/day)
   * ``renewal_rate`` - Sector renewal rate (0-1)
   * ``filplus_rate`` - FIL+ deal rate (0-1)

3. **Offline Model Data**

   Scalars:

   * ``rb_power_zero`` - Initial raw byte power
   * ``qa_power_zero`` - Initial QA power
   * ``start_vested_amt`` - Starting vested amount
   * ``zero_cum_capped_power_eib`` - Initial cumulative capped power
   * ``init_baseline_eib`` - Initial baseline (EIB)
   * ``circ_supply_zero`` - Initial circulating supply
   * ``locked_fil_zero`` - Initial locked FIL
   * ``daily_burnt_fil`` - Daily burnt FIL

   Arrays (downsampled to Mondays):

   * ``historical_raw_power_eib`` - Historical raw power (EIB)
   * ``historical_qa_power_eib`` - Historical QA power (EIB)
   * ``historical_onboarded_rb_power_pib`` - Historical onboarded RB power (PIB)
   * ``historical_onboarded_qa_power_pib`` - Historical onboarded QA power (PIB)
   * ``historical_renewed_rb_power_pib`` - Historical renewed RB power (PIB)
   * ``historical_renewed_qa_power_pib`` - Historical renewed QA power (PIB)
   * ``rb_known_scheduled_expire_vec`` - Scheduled RB expirations
   * ``qa_known_scheduled_expire_vec`` - Scheduled QA expirations
   * ``known_scheduled_pledge_release_full_vec`` - Scheduled pledge releases
   * ``burnt_fil_vec`` - Burnt FIL vector
   * ``historical_renewal_rate`` - Historical renewal rate

**Success Response** (200 OK)

.. code-block:: json

   {
     "data": {
       "raw_byte_power_averaged_over_previous_30days": 3.38,
       "renewal_rate_averaged_over_previous_30days": 0.83,
       "filplus_rate_averaged_over_previous_30days": 0.86,
       "raw_byte_power": [3.2, 3.3, 3.4, ...],
       "renewal_rate": [0.81, 0.82, 0.83, ...],
       "filplus_rate": [0.84, 0.85, 0.86, ...],
       "rb_power_zero": 22.5,
       "qa_power_zero": 35.8,
       "historical_raw_power_eib": [20.1, 20.3, 20.5, ...],
       ...
     }
   }

**Error Responses**

* **503 Service Unavailable** - Data handler not initialized
* **404 Not Found** - No historical data available
* **500 Internal Server Error** - Error retrieving historical data

Simulation
----------

POST /simulate
~~~~~~~~~~~~~~

Run a Filecoin economic forecast simulation with customizable parameters. Results are downsampled to Mondays for visualization.

**Request Model**: ``SimulationRequest``

**Request Body** (all fields optional)

.. code-block:: json

   {
     "rbp": 3.38,
     "rr": 0.83,
     "fpr": 0.86,
     "lock_target": 0.3,
     "forecast_length_days": 365,
     "sector_duration_days": 540,
     "output": ["available_supply", "network_RBP_EIB"]
   }

**Parameters**

* ``rbp`` (float or array) - Raw byte power onboarding in PIB/day. Default: smoothed historical value
* ``rr`` (float or array) - Renewal rate (0-1). Default: smoothed historical value
* ``fpr`` (float or array) - FIL+ rate (0-1). Default: smoothed historical value
* ``lock_target`` (float or array) - Target lock ratio. Default: 0.3
* ``forecast_length_days`` (int) - Forecast length in days. Default: 3650 days (10 years)
* ``sector_duration_days`` (int) - Average sector duration in days. Default: 540 days
* ``output`` (string or array of strings) - Specific output field(s) to return. If not specified, returns all fields.

**Valid Output Fields**

* ``1y_return_per_sector`` - One year return per sector
* ``1y_sector_roi`` - One year sector ROI
* ``available_supply`` - Available FIL supply
* ``capped_power_EIB`` - Capped power (EIB)
* ``circ_supply`` - Circulating supply
* ``cum_baseline_reward`` - Cumulative baseline reward
* ``cum_capped_power_EIB`` - Cumulative capped power (EIB)
* ``cum_network_reward`` - Cumulative network reward
* ``cum_simple_reward`` - Cumulative simple reward
* ``day_locked_pledge`` - Daily locked pledge
* ``day_network_reward`` - Daily network reward
* ``day_onboarded_power_QAP_PIB`` - Daily onboarded QA power (PIB)
* ``day_pledge_per_QAP`` - Daily pledge per QAP
* ``day_renewed_pledge`` - Daily renewed pledge
* ``day_renewed_power_QAP_PIB`` - Daily renewed QA power (PIB)
* ``day_rewards_per_sector`` - Daily rewards per sector
* ``days`` - Day numbers
* ``disbursed_reserve`` - Disbursed reserve
* ``full_renewal_rate`` - Full renewal rate
* ``network_QAP_EIB`` - Network QA power (EIB)
* ``network_RBP_EIB`` - Network raw byte power (EIB)
* ``network_baseline_EIB`` - Network baseline (EIB)
* ``network_gas_burn`` - Network gas burn
* ``network_locked`` - Network locked FIL
* ``network_locked_pledge`` - Network locked pledge
* ``network_locked_reward`` - Network locked reward
* ``network_time`` - Network time
* ``one_year_vest_saft`` - One year SAFT vesting
* ``qa_day_onboarded_power_pib`` - QA daily onboarded power (PIB)
* ``qa_day_renewed_power_pib`` - QA daily renewed power (PIB)
* ``qa_sched_expire_power_pib`` - QA scheduled expiring power (PIB)
* ``qa_total_power_eib`` - QA total power (EIB)
* ``rb_day_onboarded_power_pib`` - RB daily onboarded power (PIB)
* ``rb_day_renewed_power_pib`` - RB daily renewed power (PIB)
* ``rb_sched_expire_power_pib`` - RB scheduled expiring power (PIB)
* ``rb_total_power_eib`` - RB total power (EIB)
* ``six_month_vest_saft`` - Six month SAFT vesting
* ``six_year_vest_foundation`` - Six year foundation vesting
* ``six_year_vest_pl`` - Six year Protocol Labs vesting
* ``six_year_vest_saft`` - Six year SAFT vesting
* ``three_year_vest_saft`` - Three year SAFT vesting
* ``total_day_vest`` - Total daily vesting
* ``total_vest`` - Total vesting
* ``two_year_vest_saft`` - Two year SAFT vesting

**Response Model**: ``SimulationResults``

**Success Response** (200 OK)

.. code-block:: json

   {
     "input": {
       "current date": "2025-10-02",
       "forecast_length_days": 365,
       "raw_byte_power": 3.38,
       "renewal_rate": 0.83,
       "filplus_rate": 0.86
     },
     "simulation_output": {
       "available_supply": [450.5, 451.2, 452.0, ...],
       "network_RBP_EIB": [22.5, 22.6, 22.7, ...],
       "circ_supply": [580.3, 581.1, 581.9, ...],
       "network_locked": [130.2, 130.5, 130.8, ...],
       ...
     }
   }

The response contains:

* ``input`` - Metadata about the simulation run (current date, forecast length, and parameter values used)
* ``simulation_output`` - Time series arrays for all requested metrics (or all metrics if no filter specified)

All array values in ``simulation_output`` are downsampled to Mondays for efficient visualization.

**Error Responses**

* **503 Service Unavailable** - Historical data not loaded yet
* **400 Bad Request** - Invalid parameters (e.g., invalid output field names)
* **500 Internal Server Error** - Simulation failed

Error Response Format
---------------------

All error responses follow this format:

.. code-block:: json

   {
     "detail": "Error message describing what went wrong"
   }

For validation errors (422 Unprocessable Entity):

.. code-block:: json

   {
     "detail": [
       {
         "type": "validation_error_type",
         "loc": ["body", "field_name"],
         "msg": "Error message",
         "input": "invalid_value"
       }
     ]
   }
