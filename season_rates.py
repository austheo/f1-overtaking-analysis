"""
Run pass detection across full seasons and emit normalised per-race rates.

Rates are expressed as passes per lap per 100 driver pairs, so races with
different grid sizes and lengths can be compared. The opening laps are
reported as a separate phase because a start-line shuffle is a different
phenomenon from steady-state overtaking, and lumping them together is what
made the single-race Australia comparison misleading.

Output: season_pass_rates.csv (long format, one row per race per phase)

    python season_rates.py
"""

import os
import time

import fastf1
import pandas as pd

from detect_passes import detect_passes

YEARS = [2025, 2026]
CACHE_DIR = "./f1_cache"
OPENING_LAPS = 10          # boundary between "opening" and "steady" phase
OUTFILE = "season_pass_rates.csv"


def race_rows(session, year, rnd, name):
    """Return one row per phase for a single race, or [] if unusable."""
    laps = session.laps
    if laps is None or laps.empty:
        return []

    events = detect_passes(laps)
    if events.empty:
        return []

    clean = events[events["clean"]]
    n_drivers = laps["Driver"].nunique()
    pairs = n_drivers * (n_drivers - 1) / 2
    if pairs == 0:
        return []

    # Laps that actually produced a transition we could observe.
    observed = sorted(laps["LapNumber"].dropna().unique())
    if len(observed) < 2:
        return []

    phases = {
        "opening": [l for l in observed if l <= OPENING_LAPS],
        "steady": [l for l in observed if l > OPENING_LAPS],
    }

    rows = []
    for phase, phase_laps in phases.items():
        if len(phase_laps) < 2:
            continue
        n = len(clean[clean["lap"].isin(phase_laps)])
        transitions = len(phase_laps) - 1
        rows.append({
            "year": year,
            "round": rnd,
            "event": name,
            "phase": phase,
            "n_drivers": n_drivers,
            "pairs": int(pairs),
            "transitions": transitions,
            "passes": n,
            "rate_per_100_pairs": round(n / transitions / pairs * 100, 4),
            "total_events": len(events),
            "pit_excluded": int(events["pit_involved"].sum()),
            "neutral_excluded": int((events["neutralised"] != "").sum()),
            "jump_excluded": int((events["pos_gain"] > 2).sum()),
        })
    return rows


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    all_rows, skipped = [], []

    for year in YEARS:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            print(f"{year}: could not load schedule - {exc}")
            continue

        print(f"\n=== {year}: {len(schedule)} events ===")
        for _, ev in schedule.iterrows():
            rnd, name = int(ev["RoundNumber"]), ev["EventName"]
            try:
                session = fastf1.get_session(year, rnd, "R")
                session.load(telemetry=False, weather=False, messages=False)
                rows = race_rows(session, year, rnd, name)
                if rows:
                    all_rows.extend(rows)
                    tot = sum(r["passes"] for r in rows)
                    print(f"  R{rnd:2} {name[:34]:34} {tot:3} clean")
                else:
                    skipped.append((year, rnd, name, "no usable laps"))
                    print(f"  R{rnd:2} {name[:34]:34}  -- no data")
            except Exception as exc:
                skipped.append((year, rnd, name, str(exc)[:60]))
                print(f"  R{rnd:2} {name[:34]:34}  -- {str(exc)[:40]}")
            time.sleep(0.5)

    if not all_rows:
        raise SystemExit("No races produced data. Check the cache and network.")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTFILE, index=False)
    print(f"\nWrote {len(df)} rows ({df['round'].nunique()} races) to {OUTFILE}")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for y, r, n, why in skipped:
            print(f"  {y} R{r} {n[:30]} - {why}")

    print("\n=== Median rate per 100 pairs (per lap) ===")
    summary = (df.groupby(["year", "phase"])["rate_per_100_pairs"]
                 .agg(["count", "median", "mean", "std"])
                 .round(4))
    print(summary.to_string())
    print("\nMedian is the headline figure - single races with incidents "
          "skew the mean badly, as Australia showed.")


if __name__ == "__main__":
    main()
