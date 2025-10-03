MechaFil Server Documentation
==============================

**MechaFil Server** is a production-ready FastAPI service that provides HTTP endpoints for running Filecoin economic forecasts using real historical blockchain data and sophisticated simulation models.

Features
--------

* **Historical Data API**: Access to processed Filecoin network metrics
* **Simulation API**: Run economic forecasts with customizable parameters
* **Real-time Processing**: Uses live data from Spacescope for up-to-date simulations
* **Intelligent Caching**: Performance optimization with DiskCache
* **Automated Refresh**: Daily data updates at configurable times
* **Production Testing**: Comprehensive test suite validating API responses

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   cd mechafil-server
   poetry install

Configuration
~~~~~~~~~~~~~

Set up your Spacescope API credentials in a ``.env`` file:

.. code-block:: bash

   SPACESCOPE_TOKEN=Bearer YOUR_TOKEN_HERE
   # or
   SPACESCOPE_AUTH_FILE=./auths/spacescope_auth.json

Running the Server
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run mechafil-server

The server will start on ``http://localhost:8000``. Visit:

* Swagger UI: http://localhost:8000/docs
* ReDoc: http://localhost:8000/redoc

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/endpoints
   api/models
   examples/quickstart
   examples/advanced
   configuration
   deployment

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
