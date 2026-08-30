"""
Session-one verification: what can we actually see in 2026 telemetry?

Checks three things before any pipeline work begins:
  1. Does the DRS channel populate in 2025 and go dead in 2026?
  2. Are there any new/renamed telemetry channels in 2026?
  3. Does positional (X/Y) data still work in 2026?

Run:  python3 verify_2026_data.py
"""

import os
import fastf1
import pandas as pd

CIRCUIT = "Australia"
BASELINE_YEAR = 2025
TREATMENT_YEAR = 2026
SESSION = "R"
CACHE_DIR = "./f1_cache"

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race(year):
    """Load one race session with telemetry. Returns None if unavailable."""
    try:
        session = fastf1.get_session(year, CIRCUIT, SESSION)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        return session
    except Exception as exc:
        print(f"  !! could not load {year} {CIRCUIT}: {exc}")
        return None


def sample_telemetry(session, n_drivers=5):
    """Grab car telemetry for a handful of drivers, concatenated."""
    try:
        all_laps = session.laps
    except Exception as exc:
        print(f"  !! lap data never loaded: {exc}")
        return pd.DataFrame()

    frames = []
    for drv in session.drivers[:n_drivers]:
        laps = all_laps.pick_drivers(drv)
        if laps.empty:
            continue
        try:
            frames.append(laps.get_car_data())
        except Exception:
            continue
    return pd.concat(frames) if frames else pd.DataFrame()


def check_drs(tel, year):
    if "DRS" not in tel.columns:
        print(f"  {year}: no DRS column at all")
        return
    values = tel["DRS"].value_counts().sort_index()
    nonzero = (tel["DRS"] != 0).mean() * 100
    print(f"  {year}: DRS non-zero in {nonzero:.1f}% of samples")
    print(f"        distinct values -> {dict(values)}")


def check_position(session, year):
    try:
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()
        has_xy = {"X", "Y"}.issubset(pos.columns)
        spread = pos["X"].max() - pos["X"].min() if has_xy else 0
        print(f"  {year}: positional data rows={len(pos)}, X/Y present={has_xy}, X range={spread}")
    except Exception as exc:
        print(f"  {year}: positional data FAILED -> {exc}")


print("Loading sessions (first run downloads 50-100MB each)...\n")
base = load_race(BASELINE_YEAR)
treat = load_race(TREATMENT_YEAR)

if base is None or treat is None:
    raise SystemExit("Could not load both seasons. Check the circuit name and year.")

base_tel = sample_telemetry(base)
treat_tel = sample_telemetry(treat)

if base_tel.empty or treat_tel.empty:
    both_failed = base_tel.empty and treat_tel.empty
    print("\n=== Could not draw a verdict ===")
    if both_failed:
        print("  BOTH seasons returned no telemetry. That points at access, not")
        print("  at the data. F1's livetiming server returns HTTP 403 to some")
        print("  networks - datacentre and cloud IP ranges especially, which")
        print("  includes Colab and most hosted notebooks.")
        print("  Fix: run this locally on a home connection, then re-run.")
    else:
        failed = BASELINE_YEAR if base_tel.empty else TREATMENT_YEAR
        print(f"  Only {failed} returned no telemetry, while the other season")
        print("  loaded fine. That asymmetry IS a real finding - it means the")
        print("  data genuinely differs between seasons rather than being blocked.")
    raise SystemExit(1)

print("\n=== 1. DRS channel ===")
check_drs(base_tel, BASELINE_YEAR)
check_drs(treat_tel, TREATMENT_YEAR)

print("\n=== 2. Telemetry channel diff ===")
base_cols = set(base_tel.columns)
treat_cols = set(treat_tel.columns)
print(f"  shared:        {sorted(base_cols & treat_cols)}")
print(f"  only in {BASELINE_YEAR}: {sorted(base_cols - treat_cols) or 'none'}")
print(f"  only in {TREATMENT_YEAR}: {sorted(treat_cols - base_cols) or 'none'}")

print("\n=== 3. Positional data ===")
check_position(base, BASELINE_YEAR)
check_position(treat, TREATMENT_YEAR)

print("\n=== Verdict ===")
new_channels = treat_cols - base_cols
if new_channels:
    print(f"  Undocumented 2026 channels found: {sorted(new_channels)}")
    print("  Worth investigating - this is not widely known.")
else:
    print("  No new channels. Active aero state is not exposed, as expected.")
    print("  Proceed with the inference-based design.")
