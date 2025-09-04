#!/usr/bin/env python3
import argparse
import json

def main():
    p = argparse.ArgumentParser(description="Generate simulation payload JSON with vector params")
    p.add_argument("--forecast_length", type=int, default=365, help="Forecast horizon in days")
    p.add_argument("--sector_duration", type=int, default=540, help="Sector duration in days")
    p.add_argument("--rbp", type=float, default=3.3795318603515625, help="Daily onboarded RB power (PIB/day)")
    p.add_argument("--rr", type=float, default=0.834245140193526, help="Renewal rate (0..1)")
    p.add_argument("--fpr", type=float, default=0.8558804137732767, help="FIL+ rate (0..1)")
    p.add_argument("--lock_target", type=float, default=0.0, help="Target lock ratio")
    args = p.parse_args()

    payload = {
        "rbp": [args.rbp] * args.forecast_length,
        "rr": [args.rr] * args.forecast_length,
        "fpr": [args.fpr] * args.forecast_length,
        "lock_target": args.lock_target,
        "forecast_length_days": args.forecast_length,
        "sector_duration_days": args.sector_duration,
    }
    print(json.dumps(payload))

if __name__ == "__main__":
    main()

