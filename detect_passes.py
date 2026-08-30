"""
Stage A: detect position changes between drivers, lap by lap.

A "pass" here means driver A was behind driver B at the end of lap N-1
and ahead at the end of lap N. That is a candidate, not a confirmed
on-track overtake - pit cycles and retirements produce the same signal,
so each candidate is flagged rather than silently dropped.

Run on real data:   python detect_passes.py
Run the self-test:  python detect_passes.py --test
"""

import sys
from itertools import combinations

import pandas as pd

CIRCUIT = "Australia"
YEARS = [2025, 2026]
SESSION = "R"
CACHE_DIR = "./f1_cache"

# TrackStatus codes that mean racing was neutralised.
NEUTRALISED = {"4": "safety car", "5": "red flag", "6": "VSC", "7": "VSC ending"}

# A driver gaining more than this many places in one lap relative to another
# is almost never an overtake - it is the other driver having an incident,
# a spin, or a lap-time collapse. Treated as not-a-pass.
MAX_POS_GAIN = 2


def detect_passes(laps: pd.DataFrame) -> pd.DataFrame:
    """Find every lap-on-lap position swap between pairs of drivers.

    Expects the FastF1 Laps columns: LapNumber, Driver, Position,
    PitInTime, PitOutTime, TrackStatus.
    """
    grid = laps.pivot_table(index="LapNumber", columns="Driver", values="Position")

    # (driver, lap) pairs where the driver entered or exited the pits.
    pitted = set()
    for _, row in laps.iterrows():
        if pd.notna(row.get("PitInTime")) or pd.notna(row.get("PitOutTime")):
            pitted.add((row["Driver"], row["LapNumber"]))

    # Track status per lap. Codes can be concatenated, e.g. "14".
    status = (
        laps.groupby("LapNumber")["TrackStatus"]
        .agg(lambda s: "".join(sorted(set("".join(s.dropna().astype(str))))))
        .to_dict()
    )

    records = []
    lap_numbers = sorted(grid.index)

    for prev_lap, lap in zip(lap_numbers, lap_numbers[1:]):
        before, after = grid.loc[prev_lap], grid.loc[lap]
        racing = [d for d in grid.columns
                  if pd.notna(before[d]) and pd.notna(after[d])]

        for a, b in combinations(racing, 2):
            # Orient so `passer` is the one who gained.
            if before[a] > before[b] and after[a] < after[b]:
                passer, passed = a, b
            elif before[b] > before[a] and after[b] < after[a]:
                passer, passed = b, a
            else:
                continue

            pit_involved = any(
                (d, l) in pitted
                for d in (passer, passed)
                for l in (prev_lap, lap)
            )
            codes = status.get(lap, "")
            neutral = [label for c, label in NEUTRALISED.items() if c in codes]

            records.append({
                "lap": int(lap),
                "passer": passer,
                "passed": passed,
                "pos_before": int(before[passer]),
                "pos_after": int(after[passer]),
                "pos_gain": int(before[passer] - after[passer]),
                "pit_involved": pit_involved,
                "neutralised": ", ".join(neutral) or "",
                # Position is recorded at the END of a lap, so the earliest
                # transition available is lap 1 -> lap 2, an ordinary racing
                # lap. Start-line chaos happens between the grid and the end
                # of lap 1 and is invisible here. This stays False unless
                # grid positions are prepended as lap 0.
                "start_chaos": lap <= 1,
            })

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["clean"] = (
        ~out["pit_involved"]
        & (out["neutralised"] == "")
        & ~out["start_chaos"]
        & (out["pos_gain"] <= MAX_POS_GAIN)
    )
    return out.sort_values("lap").reset_index(drop=True)


# ---------------------------------------------------------------
# Self-test: synthetic laps with passes planted at known points.
# ---------------------------------------------------------------
def _synthetic_laps() -> pd.DataFrame:
    """Four drivers, five laps. Planted events:
         lap 2 - VER passes HAM   (clean)
         lap 3 - LEC passes HAM   (HAM pitted -> flagged pit_involved)
         lap 5 - SAI gains 3 places over HAM (incident, not an overtake)
    """
    positions = {
        1: {"HAM": 1, "VER": 2, "LEC": 3, "SAI": 4},
        2: {"VER": 1, "HAM": 2, "LEC": 3, "SAI": 4},
        3: {"VER": 1, "LEC": 2, "HAM": 3, "SAI": 4},
        4: {"VER": 1, "LEC": 2, "HAM": 3, "SAI": 4},
        5: {"VER": 1, "LEC": 2, "SAI": 3, "HAM": 4},
    }
    # Force a 3-place gain for SAI on lap 5 by dropping HAM down the order.
    positions[4] = {"VER": 1, "LEC": 2, "HAM": 3, "SAI": 6}
    positions[5] = {"VER": 1, "LEC": 2, "SAI": 3, "HAM": 7}

    rows = []
    for lap, order in positions.items():
        for drv, pos in order.items():
            pit = (pd.Timedelta(minutes=1)
                   if (drv == "HAM" and lap == 3) else pd.NaT)
            rows.append({
                "LapNumber": float(lap),
                "Driver": drv,
                "Position": float(pos),
                "PitInTime": pit,
                "PitOutTime": pd.NaT,
                "TrackStatus": "1",
            })
    return pd.DataFrame(rows)


def _run_tests() -> int:
    df = detect_passes(_synthetic_laps())
    print(df.to_string(index=False), "\n")

    failures = []

    def check(label, condition):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}")
        if not condition:
            failures.append(label)

    check("found at least the 3 planted events", len(df) >= 3)

    ver = df[(df.passer == "VER") & (df.passed == "HAM")]
    check("VER over HAM detected on lap 2",
          len(ver) == 1 and ver.iloc[0]["lap"] == 2)
    check("VER over HAM marked clean",
          len(ver) == 1 and bool(ver.iloc[0]["clean"]))

    lec = df[(df.passer == "LEC") & (df.passed == "HAM")]
    check("LEC over HAM detected on lap 3",
          len(lec) == 1 and lec.iloc[0]["lap"] == 3)
    check("LEC over HAM flagged as pit-related",
          len(lec) == 1 and bool(lec.iloc[0]["pit_involved"]))
    check("LEC over HAM excluded from clean",
          len(lec) == 1 and not bool(lec.iloc[0]["clean"]))

    check("no phantom pass between VER and LEC",
          df[(df.passer == "LEC") & (df.passed == "VER")].empty)

    jump = df[(df.passer == "SAI") & (df.passed == "HAM") & (df.lap == 5)]
    check("SAI 3-place gain over HAM is recorded", len(jump) == 1)
    check("SAI 3-place gain excluded from clean",
          len(jump) == 1 and not bool(jump.iloc[0]["clean"]))
    check("every clean event gained <= 2 places",
          df[df["clean"]]["pos_gain"].max() <= 2)

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


def _run_real() -> int:
    import os
    import fastf1

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    for year in YEARS:
        session = fastf1.get_session(year, CIRCUIT, SESSION)
        session.load(telemetry=False, weather=False, messages=False)

        events = detect_passes(session.laps)
        if events.empty:
            print(f"\n{year}: no position changes found - check the data loaded.")
            continue

        clean = events[events["clean"]]
        print(f"\n=== {year} {CIRCUIT} ===")
        print(f"  {len(events)} position changes, {len(clean)} clean")
        print(f"  excluded: {events['pit_involved'].sum()} pit-related, "
              f"{(events['neutralised'] != '').sum()} under neutralisation, "
              f"{events['start_chaos'].sum()} at the start")
        print("\n  Clean passes:")
        print(clean[["lap", "passer", "passed", "pos_before", "pos_after"]]
              .to_string(index=False))

        events.to_csv(f"passes_{year}_{CIRCUIT}.csv", index=False)
        print(f"\n  written to passes_{year}_{CIRCUIT}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(_run_tests() if "--test" in sys.argv else _run_real())
