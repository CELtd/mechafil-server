from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Union


@dataclass
class SimulationResults:
    """
    Container for the results of a Filecoin economic simulation.

    The object has two main fields:

    - `input_data`: dictionary containing metadata about the simulation run:
        * "current date": current chain date when simulation starts (string, YYYY-MM-DD)
        * "forecast_length_days": length of forecast horizon (int)
        * "raw_byte_power": RBP value used for the simulation (float)
        * "renewal_rate": RR used for the simulation (float)
        * "filplus_rate": FIL+ rate used for the simulation (float)

    - `simulation_output`: dictionary mapping metric names to time series
      arrays or scalars. The keys include, among others:
        * "available_supply"
        * "capped_power_EIB"
        * "circ_supply"
        * "cum_baseline_reward"
        * "cum_capped_power_EIB"
        * "cum_network_reward"
        * "cum_simple_reward"
        * "day_locked_pledge"
        * "day_network_reward"
        * "day_onboarded_power_QAP_PIB"
        * "day_pledge_per_QAP"
        * "day_renewed_pledge"
        * "day_renewed_power_QAP_PIB"
        * "day_rewards_per_sector"
        * "days"
        * "network_QAP_EIB"
        * "network_RBP_EIB"
        * "network_baseline_EIB"
        * "network_locked"
        * "qa_total_power_eib"
        * "rb_total_power_eib"
        * "six_year_vest_saft"
        * "three_year_vest_saft"
        * "two_year_vest_saft"

      By convention:
        - Most values are lists of floats, trimmed to `forecast_length_days`.
        - "1y_return_per_sector" and "1y_sector_roi" are special: they are 90 days shorter.
        - Scalars (e.g., initial pledge, some vesting totals) are returned as floats.
    """ 
    input_data: Dict[str, Any]
    simulation_output: Dict[str, Union[List[float], float, str]]

    @classmethod
    def from_raw(
        cls,
        raw_results: Dict[str, Any],
        start_date,
        current_date,
        forecast_len,
        smoothed_rbp,
        smoothed_rr,
        smoothed_fpr
    ) -> "SimulationResults":
        diff = (current_date - start_date).days if hasattr(current_date, "__sub__") else None

        input_data = {
            "current date": current_date.strftime("%Y-%m-%d") if hasattr(current_date, "strftime") else current_date,
            "forecast_length_days": forecast_len,
            "raw_byte_power": round(float(smoothed_rbp), 2),
            "renewal_rate": round(float(smoothed_rr), 2),
            "filplus_rate": round(float(smoothed_fpr), 2),
        }

        simulation_output = {}
        for k, v in raw_results.items():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                arr = [round(float(item), 2) for item in v]
                if k not in ("1y_return_per_sector", "1y_sector_roi"):
                    if len(arr) > forecast_len and diff is not None:
                        arr = arr[diff: diff + forecast_len]
                    if len(arr) > forecast_len:
                        arr = arr[:forecast_len]
                simulation_output[k] = arr
            elif isinstance(v, (int, float)):
                simulation_output[k] = round(float(v), 2)
            else:
                simulation_output[k] = v

        return cls(input_data=input_data, simulation_output=simulation_output)

    def downsample_mondays(self, start_date: date) -> "SimulationResults":
        """
        Return a new SimulationResults object with arrays downsampled to Mondays.
        """
        def select_mondays(data_array, start_date):
            mondays = []
            for i, val in enumerate(data_array):
                current = start_date + timedelta(days=i)
                if current.weekday() == 0:  # Monday
                    mondays.append(round(float(val), 2))
            return mondays

        downsampled = {}
        for k, v in self.simulation_output.items():
            if isinstance(v, list) and len(v) > 1:
                downsampled[k] = select_mondays(v, start_date)
            else:
                downsampled[k] = v

        return SimulationResults(
            input_data=self.input_data.copy(),
            simulation_output=downsampled
        )

    def filter_fields(self, fields: Union[str, List[str]]) -> "SimulationResults":
        """
        Return a new SimulationResults with only the requested fields
        included in simulation_output.
        """
        if isinstance(fields, str):
            fields = [fields]

        filtered = {f: self.simulation_output.get(f) for f in fields if f in self.simulation_output}
        missing = [f for f in fields if f not in self.simulation_output]
        if missing:
            # you can decide whether to raise or just warn
            import logging
            logging.warning(f"Requested fields not found in simulation results: {missing}")

        return SimulationResults(
            input_data=self.input_data.copy(),
            simulation_output=filtered
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for JSON responses)."""
        return {
            "input": self.input_data,
            "simulation_output": self.simulation_output
        }

@dataclass
class FetchDataResults:
    """
    Container for complete historical Filecoin data returned by the /historical-data endpoint.

    The output is provided in a single dictionary under the field `data`.  
    It includes three categories of fields:

    1. **30-day averaged metrics (scalars)**:
       - `raw_byte_power_averaged_over_previous_30days` (float)
       - `renewal_rate_averaged_over_previous_30days` (float)
       - `filplus_rate_averaged_over_previous_30days` (float)

    2. **Daily historical arrays** (lists of floats, one entry per day):
       - `raw_byte_power`
       - `renewal_rate`
       - `filplus_rate`

    3. **Offline model data (scalars and arrays)**:
       - Scalars:
         * `rb_power_zero`, `qa_power_zero`
         * `start_vested_amt`, `zero_cum_capped_power_eib`
         * `init_baseline_eib`, `circ_supply_zero`
         * `locked_fil_zero`, `daily_burnt_fil`
       - Arrays:
         * `historical_raw_power_eib`, `historical_qa_power_eib`
         * `historical_onboarded_rb_power_pib`, `historical_onboarded_qa_power_pib`
         * `historical_renewed_rb_power_pib`, `historical_renewed_qa_power_pib`
         * `rb_known_scheduled_expire_vec`, `qa_known_scheduled_expire_vec`
         * `known_scheduled_pledge_release_full_vec`
         * `burnt_fil_vec`
         * `historical_renewal_rate`

    All scalars are stored as floats, arrays as lists of floats (rounded to 6 decimals).  

    Utility methods:
      - `.filter_fields(fields)`: return a new FetchDataResults with only a subset of keys.
      - `.to_dict()`: produce a JSON-serializable representation with the `data` field.
    """
    data: Dict[str, Union[List[float], float, str]]

    @classmethod
    def from_raw(
        cls,
        hist_arrays: Dict[str, Any],
        offline_data: Dict[str, Any],
        smoothed_rbp: float,
        smoothed_rr: float,
        smoothed_fpr: float
    ) -> "FetchDataResults":
        """
        Build FetchDataResults from the raw historical arrays,
        offline data, and smoothed metrics.
        """
        combined_data = {}
        # Smoothed metrics -> represent as "averaged over previous 30 days"
        combined_data["raw_byte_power_averaged_over_previous_30days"] = round(float(smoothed_rbp), 6)
        combined_data["renewal_rate_averaged_over_previous_30days"] = round(float(smoothed_rr), 6)
        combined_data["filplus_rate_averaged_over_previous_30days"] = round(float(smoothed_fpr), 6)

        # Historical arrays
        for k, v in hist_arrays.items():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                combined_data[k] = [round(float(x), 6) for x in v]
            elif isinstance(v, (int, float)):
                combined_data[k] = round(float(v), 6)
            else:
                combined_data[k] = v

        # Offline data
        for k, v in offline_data.items():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                combined_data[k] = [round(float(x), 6) for x in v]
            elif isinstance(v, (int, float)):
                combined_data[k] = round(float(v), 6)
            else:
                combined_data[k] = v

        return cls(data=combined_data)

    def downsample_mondays(self, start_date: date) -> "FetchDataResults":
        """
        Return a new FetchDataResults object with arrays downsampled to Mondays.

        Args:
            start_date: The starting date for the historical data arrays

        Returns:
            New FetchDataResults with Monday-only values for array fields
        """
        def select_mondays(data_array, start_date):
            mondays = []
            for i, val in enumerate(data_array):
                current = start_date + timedelta(days=i)
                if current.weekday() == 0:  # Monday
                    mondays.append(round(float(val), 6))
            return mondays

        downsampled = {}
        for k, v in self.data.items():
            if isinstance(v, list) and len(v) > 1:
                downsampled[k] = select_mondays(v, start_date)
            else:
                # Keep scalars and single-element arrays unchanged
                downsampled[k] = v

        return FetchDataResults(data=downsampled)

    def filter_fields(self, fields: Union[str, List[str]]) -> "FetchDataResults":
        """
        Return a new FetchDataResults with only the requested fields.
        """
        if isinstance(fields, str):
            fields = [fields]

        filtered = {f: self.data.get(f) for f in fields if f in self.data}
        missing = [f for f in fields if f not in self.data]
        if missing:
            import logging
            logging.warning(f"Requested fields not found in fetch results: {missing}")

        return FetchDataResults(data=filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for JSON responses)."""
        return {"data": self.data}
