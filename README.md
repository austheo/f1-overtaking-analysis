# Did F1's 2026 rules actually improve overtaking?

**A decision-timing and treatment-effect problem — the same structure as any
before/after policy evaluation where the intervention changed several things at
once. Built on a dataset assembled from scratch, including a channel the sport
stopped publishing.**

---

## The question

For fourteen seasons the Drag Reduction System did two jobs at once: it gave a
car an aerodynamic advantage, and it gated that advantage behind a proximity
condition — you had to be within one second of the car ahead.

The 2026 regulations split those jobs apart:

| | 2011–2025 | 2026 onwards |
|---|---|---|
| Aerodynamic advantage | DRS, rear wing only | **Straight Mode** — front and rear wings, available to every car |
| Proximity condition | Within 1s to use DRS | **Overtake Mode** — extra energy for a car within 1s |

So the aero help became universal and unconditional, while the chaser-specific
advantage moved from the wing to the power unit.

**If that unbundling matters, two things should follow.** Overtaking should shift
in character rather than simply increase. And passes should become less
spatially concentrated, because DRS was locked to fixed zones while energy
deployment is not.

This analysis tests both.

---

## The constraint that shaped the project

F1 does not publish active-aero or energy-deployment state. FastF1's source
carries the comment `# drs is no longer included in 2026`, and the maintainer
has confirmed the TV-graphics data is not available through any public tool.

Verified directly on 2025 vs 2026 Australian GP telemetry:

| Season | DRS samples | Active (codes 10/12/14) |
|---|---|---|
| 2025 | 114,245 | 1,427 (1.25%) |
| 2026 | 92,803 | **0** |

The channel is not sparse in 2026. It is dead.

This rules out any design that compares DRS activations to Straight Mode
activations. The project therefore measures **outcomes** — how much overtaking
happened, and where — rather than mechanism.

---

## Findings

### 1. Opening-lap overtaking roughly doubled — robust

Passes per lap per 100 driver pairs, paired across 11 circuits raced in both
seasons:

| Phase | 2025 median | 2026 median | Circuits improved | Wilcoxon p |
|---|---|---|---|---|
| Opening (laps 1–10) | 0.130 | 0.611 | 9 / 11 | **0.0039** |
| Steady (laps 11+) | 0.257 | 0.178 | 6 / 11 | 0.83 |

Four circuits had a 2025 opening rate of exactly zero (safety-car starts).
Excluding them: still 6 of 7 improved, p = 0.031, median ratio 2.06×.

The pattern also survives restricting 2025 to rounds 1–12, ruling out an
early-season artefact.

### 2. Steady-state overtaking is unchanged — null

An earlier unmatched comparison suggested a 31% decline. It did not survive
circuit matching (p = 0.83, sign slightly positive). **Not a finding.**

### 3. Pass locations are no more dispersed — null, after correction

581 of 650 detected passes (89%) were located on track by finding where the two
drivers' distance-along-lap crossed. Dispersion measured as normalised entropy
over the lap, 10 circuits (Monaco excluded for missing 2026 telemetry).

| Analysis | Circuits with higher dispersion | Wilcoxon p |
|---|---|---|
| Raw | 9 / 10 | 0.0059 |
| **Rarefied to equal n** | **4 / 10** | **0.557** |

Entropy is biased upward by sample size, and 2026 had more passes at 8 of 10
circuits. Subsampling each year to equal n removes the effect entirely. The raw
result was an artefact.

---

## What went wrong, and how it was caught

This section exists because the mistakes were more instructive than the results.

**A silent event substitution.** FastF1's fuzzy matcher resolved the string
`"Great Britain"` to the *Austrian* Grand Prix without raising an error. Austria
was counted twice and Silverstone was absent. Nothing failed; the numbers simply
looked plausible. The loader now verifies the returned event name against the
request and skips on mismatch.

**A filter that excluded real data.** Lap-1 position changes were initially
discarded as "start chaos". But position is recorded at the *end* of a lap, so
the first available transition is lap 1 → 2 — an ordinary racing lap. The filter
removed genuine passes while missing the phenomenon it targeted. Caught by a
synthetic test with planted events.

**A significant result that wasn't.** See finding 3. The raw entropy comparison
returned p = 0.0059 in the direction the theory predicted. It was sample-size
bias. Rarefaction was the only thing standing between a plausible false positive
and the write-up.

Detection logic is covered by synthetic-fixture tests with known planted events
(`--test` on both scripts), because the real data has no ground truth to check
against.

---

## Limitations

- **No causal identification.** The 2026 package changed aerodynamics, power
  units, tyres, car weight and the grid simultaneously. Nothing here isolates
  Straight Mode or Overtake Mode from the rest.
- **Small samples.** 11 paired circuits; the 2026 season was 12 of 23 rounds
  complete at time of analysis.
- **Undercounting.** A pass and repass within the same lap nets to zero position
  change and is invisible to the method. This systematically misses the closest
  racing.
- **Threshold choice.** Position gains above 2 places are treated as incidents
  rather than overtakes. This is a judgement call, not a principled cutoff.
- **Unresolved events.** 11% of passes could not be located on track (lapped
  traffic, start-line straddling, telemetry gaps). Monaco 2026 has no telemetry
  at all.

---

## Running it

```bash
pip install fastf1
python verify_2026_data.py     # confirms what telemetry is available
python detect_passes.py --test # synthetic tests
python season_rates.py         # per-race normalised pass rates
python locate_passes.py        # where on track passes occur
```

Requires a home network connection — F1's timing server returns HTTP 403 to
datacentre IP ranges, so hosted notebooks (Colab and similar) will silently
return empty data.

| File | Purpose |
|---|---|
| `verify_2026_data.py` | Confirms channel availability across seasons |
| `detect_passes.py` | Lap-on-lap position swaps, with exclusion flags |
| `season_rates.py` | Normalises to passes per lap per 100 driver pairs |
| `locate_passes.py` | Locates each pass as a fraction of lap distance |

---

## Glossary

**Drag Reduction System (2011–2025)** — A driver-activated flap on the rear
wing, usable only in designated zones and only when within one second of the car
ahead at a detection point.

**Straight Mode (2026–)** — Active aerodynamics on both front and rear wings,
open on defined straights. Unlike DRS, available to every car regardless of the
gap ahead. Closes automatically on braking or lifting off.

**Overtake Mode (2026–)** — Additional electrical deployment for a car within
one second of the car ahead at an activation point. This, not Straight Mode, is
the functional successor to DRS's proximity condition.

---

*Data: FastF1 3.8.3, which sources results and driver metadata from the Jolpica
API. Analysis in Python (pandas, scipy).*
