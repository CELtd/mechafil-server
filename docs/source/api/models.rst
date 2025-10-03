Data Models
===========

This page documents the Pydantic data models used for request and response validation.

Request Models
--------------

SimulationRequest
~~~~~~~~~~~~~~~~~

Request model for the ``/simulate`` endpoint.

**Fields**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Field
     - Type
     - Required
     - Description
   * - ``rbp``
     - float | List[float]
     - No
     - Raw byte power onboarding (PIB/day), constant or array. Default: smoothed historical value
   * - ``rr``
     - float | List[float]
     - No
     - Renewal rate (0-1), constant or array. Default: smoothed historical value
   * - ``fpr``
     - float | List[float]
     - No
     - FIL+ rate (0-1), constant or array. Default: smoothed historical value
   * - ``lock_target``
     - float | List[float]
     - No
     - Target lock ratio (e.g., 0.3), float or array. Default: 0.3
   * - ``forecast_length_days``
     - int
     - No
     - Forecast length in days. Default: 3650 (10 years)
   * - ``sector_duration_days``
     - int
     - No
     - Average sector duration in days. Default: 540
   * - ``output``
     - str | List[str]
     - No
     - Specific output field(s) to return. If not specified, returns all fields

**Example**

.. code-block:: json

   {
     "rbp": 3.3795318603515625,
     "rr": 0.834245140193526,
     "fpr": 0.8558804137732767,
     "lock_target": 0.3,
     "forecast_length_days": 365,
     "sector_duration_days": 540,
     "output": ["available_supply", "network_RBP_EIB"]
   }

**Validation**

* ``output`` field values must be valid simulation output field names
* Invalid field names will result in a 422 validation error with the list of valid fields

Response Models
---------------

HealthResponse
~~~~~~~~~~~~~~

Response model for the ``/health`` endpoint.

**Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``status``
     - str
     - Service status (e.g., "healthy")
   * - ``version``
     - str
     - Service version (e.g., "0.1.0")
   * - ``jax_backend``
     - str | None
     - JAX backend in use (e.g., "cpu", "gpu")

**Example**

.. code-block:: json

   {
     "status": "healthy",
     "version": "0.1.0",
     "jax_backend": "cpu"
   }

SimulationResults
~~~~~~~~~~~~~~~~~

Response model for the ``/simulate`` endpoint.

**Structure**

The response contains two main sections:

1. ``input`` - Metadata about the simulation run
2. ``simulation_output`` - Time series data and metrics

**Input Metadata Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``current date``
     - str
     - Current chain date when simulation starts (YYYY-MM-DD)
   * - ``forecast_length_days``
     - int
     - Length of forecast horizon in days
   * - ``raw_byte_power``
     - float
     - RBP value used for simulation (rounded to 2 decimals)
   * - ``renewal_rate``
     - float
     - Renewal rate used for simulation (rounded to 2 decimals)
   * - ``filplus_rate``
     - float
     - FIL+ rate used for simulation (rounded to 2 decimals)

**Simulation Output Fields**

The ``simulation_output`` object contains time series arrays for various metrics. All arrays are downsampled to Mondays for efficient visualization, with values rounded to 2 decimal places.

Available fields (when no filter is applied):

* Power Metrics:

  * ``network_RBP_EIB`` - Network raw byte power (EIB)
  * ``network_QAP_EIB`` - Network quality-adjusted power (EIB)
  * ``network_baseline_EIB`` - Network baseline (EIB)
  * ``capped_power_EIB`` - Capped power (EIB)
  * ``cum_capped_power_EIB`` - Cumulative capped power (EIB)

* Supply Metrics:

  * ``circ_supply`` - Circulating supply
  * ``available_supply`` - Available supply
  * ``network_locked`` - Total network locked FIL
  * ``network_locked_pledge`` - Network locked pledge
  * ``network_locked_reward`` - Network locked rewards

* Reward Metrics:

  * ``day_network_reward`` - Daily network rewards
  * ``cum_network_reward`` - Cumulative network rewards
  * ``cum_baseline_reward`` - Cumulative baseline rewards
  * ``cum_simple_reward`` - Cumulative simple rewards
  * ``day_rewards_per_sector`` - Daily rewards per sector

* Onboarding Metrics:

  * ``day_onboarded_power_QAP_PIB`` - Daily onboarded QA power (PIB)
  * ``day_renewed_power_QAP_PIB`` - Daily renewed QA power (PIB)
  * ``day_locked_pledge`` - Daily locked pledge
  * ``day_renewed_pledge`` - Daily renewed pledge
  * ``day_pledge_per_QAP`` - Daily pledge per QAP

* ROI Metrics:

  * ``1y_return_per_sector`` - One year return per sector
  * ``1y_sector_roi`` - One year sector ROI

* Vesting Metrics:

  * ``total_vest`` - Total vesting
  * ``total_day_vest`` - Total daily vesting
  * ``one_year_vest_saft`` - One year SAFT vesting
  * ``two_year_vest_saft`` - Two year SAFT vesting
  * ``three_year_vest_saft`` - Three year SAFT vesting
  * ``six_year_vest_saft`` - Six year SAFT vesting
  * ``six_month_vest_saft`` - Six month SAFT vesting
  * ``six_year_vest_foundation`` - Six year foundation vesting
  * ``six_year_vest_pl`` - Six year Protocol Labs vesting

* Other Metrics:

  * ``days`` - Day numbers
  * ``network_time`` - Network time
  * ``network_gas_burn`` - Network gas burn
  * ``disbursed_reserve`` - Disbursed reserve
  * ``full_renewal_rate`` - Full renewal rate
  * Various raw/QA power breakdown metrics

**Example Response**

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
       "available_supply": [450.5, 451.2, 452.0],
       "network_RBP_EIB": [22.5, 22.6, 22.7],
       "circ_supply": [580.3, 581.1, 581.9],
       "network_locked": [130.2, 130.5, 130.8]
     }
   }

FetchDataResults
~~~~~~~~~~~~~~~~

Response model for the ``/historical-data`` endpoint.

**Structure**

The response contains a single ``data`` object with all historical metrics.

**Data Fields Categories**

1. **30-day Averaged Metrics** (scalars, rounded to 6 decimals)

   * ``raw_byte_power_averaged_over_previous_30days``
   * ``renewal_rate_averaged_over_previous_30days``
   * ``filplus_rate_averaged_over_previous_30days``

2. **Historical Time Series** (arrays, downsampled to Mondays, rounded to 6 decimals)

   * ``raw_byte_power`` - Daily raw byte power onboarding
   * ``renewal_rate`` - Daily renewal rate
   * ``filplus_rate`` - Daily FIL+ rate

3. **Offline Model Data**

   Scalars (rounded to 6 decimals):

   * ``rb_power_zero`` - Initial raw byte power
   * ``qa_power_zero`` - Initial QA power
   * ``start_vested_amt`` - Starting vested amount
   * ``zero_cum_capped_power_eib`` - Initial cumulative capped power (EIB)
   * ``init_baseline_eib`` - Initial baseline (EIB)
   * ``circ_supply_zero`` - Initial circulating supply
   * ``locked_fil_zero`` - Initial locked FIL
   * ``daily_burnt_fil`` - Daily burnt FIL

   Arrays (downsampled to Mondays, rounded to 6 decimals):

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

**Example Response**

.. code-block:: json

   {
     "data": {
       "raw_byte_power_averaged_over_previous_30days": 3.379532,
       "renewal_rate_averaged_over_previous_30days": 0.834245,
       "filplus_rate_averaged_over_previous_30days": 0.855880,
       "raw_byte_power": [3.2, 3.3, 3.4, 3.35, 3.38],
       "renewal_rate": [0.81, 0.82, 0.83, 0.83, 0.83],
       "filplus_rate": [0.84, 0.85, 0.86, 0.855, 0.856],
       "rb_power_zero": 22.5,
       "qa_power_zero": 35.8,
       "historical_raw_power_eib": [20.1, 20.3, 20.5, 20.7, 20.9]
     }
   }

ErrorResponse
~~~~~~~~~~~~~

Generic error response model used across endpoints.

**Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``status``
     - str
     - Error status (default: "error")
   * - ``message``
     - str
     - Error message
   * - ``detail``
     - Dict[str, Any] | None
     - Additional error details (optional)

**Example**

.. code-block:: json

   {
     "status": "error",
     "message": "Simulation failed",
     "detail": {
       "reason": "Invalid parameter combination",
       "parameter": "forecast_length_days"
     }
   }

Data Types Reference
--------------------

Common Types
~~~~~~~~~~~~

* **EIB** - Exbibytes (2^60 bytes)
* **PIB** - Pebibytes (2^50 bytes)
* **FIL** - Filecoin tokens
* **Rate** - Decimal value between 0 and 1 (e.g., 0.83 = 83%)
* **Days** - Integer number of days
* **Power** - Storage power in EIB or PIB units

Rounding Precision
~~~~~~~~~~~~~~~~~~

* Simulation output values: 2 decimal places
* Historical data values: 6 decimal places
* Input parameters are preserved as provided
