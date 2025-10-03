from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Union


@dataclass
class SimulationResults:
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
