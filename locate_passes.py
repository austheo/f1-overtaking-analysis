"""
Stage B: find WHERE on track each detected pass happened.

For each clean pass from Stage A, pull both drivers' telemetry for that lap,
align them on session time, and find the moment the passer's distance-along-lap
overtakes the passed driver's. Reports position as a fraction of lap distance
(0 = start line, 1 = start line again) so circuits of different lengths are
directly comparable.

Start with two or three circuits to validate before scaling up - each session
downloads 50-100MB because telemetry is required here.

    python locate_passes.py --test     # synthetic check, no download
    python locate_passes.py            # real run
"""

import os
import sys

import pandas as pd

from detect_passes import detect_passes

# Exact event names. FastF1's fuzzy matcher will silently substitute a
# different event for an unrecognised string, so these must be right and
# the match is verified at load time.
CIRCUITS = [
    "Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix",
    "Miami Grand Prix", "Canadian Grand Prix", "Monaco Grand Prix",
    "Austrian Grand Prix", "British Grand Prix", "Belgian Grand Prix",
    "Hungarian Grand Prix", "Dutch Grand Prix",
]
YEARS = [2025, 2026]
CACHE_DIR = "./f1_cache"
OUTFILE = "pass_locations.csv"

# If the two drivers' aligned samples never cross, or cross more than
# this many times, the event is ambiguous and gets reported as unresolved.
MAX_CROSSINGS = 3


def find_crossing(tel_a: pd.DataFrame, tel_b: pd.DataFrame):
    """Align two drivers' telemetry and find where A passes B.

    Both frames need SessionTime and RelativeDistance columns.
    Returns (relative_distance, n_crossings) or (None, n_crossings).
    """
    a = tel_a[["SessionTime", "RelativeDistance"]].sort_values("SessionTime")
    b = tel_b[["SessionTime", "RelativeDistance"]].sort_values("SessionTime")
    if a.empty or b.empty:
        return None, 0

    merged = pd.merge_asof(
        a, b, on="SessionTime", suffixes=("_a", "_b"), direction="nearest"
    ).dropna()
    if len(merged) < 2:
        return None, 0

    # Positive once A is further round the lap than B.
    delta = merged["RelativeDistance_a"] - merged["RelativeDistance_b"]
    sign = delta.apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
    flips = sign.diff().fillna(0) != 0
    n_crossings = int(flips.sum())

    # We want the last transition from behind to ahead.
    behind_to_ahead = merged[(sign.shift(1) < 0) & (sign > 0)]
    if behind_to_ahead.empty:
        return None, n_crossings

    return float(behind_to_ahead.iloc[-1]["RelativeDistance_a"]), n_crossings


def locate_for_session(session, year, circuit):
    events = detect_passes(session.laps)
    if events.empty:
        return []

    clean = events[events["clean"]]
    rows = []

    for _, ev in clean.iterrows():
        lap_n = ev["lap"]
        try:
            la = session.laps.pick_drivers(ev["passer"]).pick_laps(lap_n)
            lb = session.laps.pick_drivers(ev["passed"]).pick_laps(lap_n)
            if la.empty or lb.empty:
                raise ValueError("missing lap")
            ta = la.get_telemetry().add_relative_distance()
            tb = lb.get_telemetry().add_relative_distance()
            loc, n_cross = find_crossing(ta, tb)
        except Exception as exc:
            loc, n_cross = None, -1
            print(f"    lap {lap_n} {ev['passer']}>{ev['passed']}: {str(exc)[:40]}")

        resolved = loc is not None and 0 <= n_cross <= MAX_CROSSINGS
        rows.append({
            "year": year,
            "circuit": circuit,
            "lap": lap_n,
            "passer": ev["passer"],
            "passed": ev["passed"],
            "rel_distance": round(loc, 4) if loc is not None else None,
            "n_crossings": n_cross,
            "resolved": resolved,
        })
    return rows


def _run_tests() -> int:
    """Synthetic: B leads until 60% of the lap, then A gets ahead."""
    import numpy as np

    t = pd.to_timedelta(np.linspace(0, 90, 200), unit="s")
    frac = np.linspace(0, 1, 200)
    # B leads by a gap that shrinks linearly to zero at 60% of the lap,
    # so A is behind before 0.6 and ahead after it.
    cross_at = 0.6
    gap = 0.02 * (1 - frac / cross_at)
    a = pd.DataFrame({"SessionTime": t, "RelativeDistance": frac})
    b = pd.DataFrame({"SessionTime": t, "RelativeDistance": frac + gap})

    loc, n = find_crossing(a, b)
    failures = []

    def check(label, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            failures.append(label)

    check("a crossing was found", loc is not None)
    check("crossing located near 60% of lap",
          loc is not None and 0.55 < loc < 0.65)
    check("exactly one crossing detected", n == 1)

    # No crossing: A stays behind the whole lap.
    b2 = pd.DataFrame({"SessionTime": t, "RelativeDistance": frac + 0.05})
    loc2, n2 = find_crossing(a, b2)
    check("no false positive when never passing", loc2 is None)

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


def _run_real() -> int:
    import fastf1

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    all_rows = []
    for year in YEARS:
        for circuit in CIRCUITS:
            print(f"\n{year} {circuit}")
            try:
                session = fastf1.get_session(year, circuit, "R")
                session.load(weather=False, messages=False)   # telemetry ON

                # Guard against silent fuzzy substitution: FastF1 will happily
                # resolve an unrecognised name to a completely different event.
                actual = str(session.event.get("EventName", ""))
                if actual.strip().lower() != circuit.strip().lower():
                    print(f"  SKIPPED: asked for '{circuit}' but got "
                          f"'{actual}' - name did not match")
                    continue

                rows = locate_for_session(session, year, circuit)
            except Exception as exc:
                print(f"  skipped: {str(exc)[:70]}")
                continue

            if not rows:
                print("  no clean passes")
                continue
            ok = sum(r["resolved"] for r in rows)
            print(f"  {len(rows)} passes, {ok} located on track")
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("Nothing located. Check the circuit names and cache.")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTFILE, index=False)
    print(f"\nWrote {len(df)} rows to {OUTFILE}")

    res = df[df["resolved"]]
    print(f"\nResolved {len(res)}/{len(df)} "
          f"({len(res)/len(df)*100:.0f}%)")
    print("\n=== Where passes happen (fraction of lap) ===")
    print(res.groupby("year")["rel_distance"]
             .describe()[["count", "mean", "50%", "std"]].round(3).to_string())
    print("\nA low standard deviation means passes cluster at a few points on")
    print("the lap. DRS-era passes should cluster at zone ends; if 2026 is")
    print("more dispersed, that is the mechanism showing up in the data.")
    return 0


if __name__ == "__main__":
    sys.exit(_run_tests() if "--test" in sys.argv else _run_real())
