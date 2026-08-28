# Scout report — 2026-08-28 — third-real-trace candidates

Produced by the `web-researcher` scout when evaluating whether to add a
third real production trace to the paper (Azure PdM synthetic +
Alibaba real per-job as of now). User specifically asked about
automotive telematics / DTC / OBD-II datasets.

## Headline

**No public automotive-DTC dataset exists at "third real trace" scale
with fleet coverage AND failure labels.** Kaggle OBD-II uploads are
lookup tables or single-vehicle traces; the Zenodo automotive-faults
record is a 61 KB JSON toy. The closest real automotive dataset is
SCANIA Component X (Nature Sci. Data 2025, 33,641 trucks) but its
"events" are histogram / counter snapshots, not DTC codes.

## Part A: automotive DTC / OBD-II candidates (all inspected)

| Candidate | Scale | Discrete DTC? | Failure label? | Verdict |
|---|---|---|---|---|
| SCANIA Component X (Nature Sci. Data 2025, DOI 10.5878/jvb5-d390) | 33,641 trucks; 1,122,452 readouts; histograms + counters + categoricals | No (histograms/counters, not DTC codes) | Yes (repaired vs healthy, 5 pre-failure time-classes) | Marginal — real production, huge, CC-BY-4.0, needs binning |
| APS Failure at Scania Trucks (UCI 421) | 60k train + 16k test rows, 171 anon attrs | No (cross-sectional, not sequences) | Binary APS failure | Not suitable |
| Kaggle OBD2 Powertrain Codes | reference table of P-codes | No sequences | None | Not suitable |
| Zenodo Automotive Faults 2025-06 | 61 kB JSON + 4 PNGs | No timestamps | No | Not suitable, toy |
| NHTSA ODI Complaints | millions of textual complaints, DateOfIncident | Free-text only | Recall linkage possible | Not suitable without heavy NLP |
| Ford Multi-AV Seasonal | 6 AVs, 2017-18 | Lidar/camera/GNSS only | No | Not suitable |
| PHM Society challenges 2022-2026 | Gear / bearing / filtration RUL | Mechanical, not DTC | Yes | No automotive DTC release |
| Kaggle Vehicle Maintenance Telemetry | small telemetry table | Not DTC events | Component failure | Marginal, unclear provenance |
| Bosch Production Line Performance | ~1.18 M assembly parts | Manufacturing events, not automotive DTC | Binary defect | Marginal, non-automotive |

## Part B: adjacent real predictive-maintenance datasets

| Candidate | Scale | Discrete events? | Access |
|---|---|---|---|
| **Loghub BGL (Blue Gene/L, LLNL)** | 4,747,963 messages, 376 event templates, 7.34% abnormal, native `alert / -` label | Yes, natively | Free, GitHub logpai/loghub |
| **Loghub HDFS_v1** | 11.2 M messages, 29 templates, 16,838 sessions, 2.9% anomalous | Yes | Free, same repo |
| **Loghub Thunderbird** | ~200 M messages, 0.49% abnormal | Yes | Free, same repo |
| Google Borg cluster-trace 2019 v3 | 8 cells, May 2019, ~2.4 TiB | Yes (SUBMIT/SCHEDULE/EVICT/FAIL/KILL) | Free via BigQuery/HTTPS |
| **Backblaze Drive Stats** | ~300k+ drives, daily SMART snapshots since 2013 | Discrete-ish (daily rows, SMART vectors) | Free, quarterly ZIPs |
| NASA C-MAPSS / N-CMAPSS | Simulated turbofan run-to-failure | No (continuous sensors) | Public NASA PCoE |

## Part C: scout recommendation (verbatim)

1. **Loghub BGL** — 4.7M discrete event-template messages, real
   production LLNL supercomputer, native `alert / -` labels. Exact
   shape our pipeline was built for. Already cited via Ren et al. 2021
   in our Related Work. No discretisation, no vocabulary construction.

2. **Backblaze Drive Stats** — real per-entity longitudinal event
   stream with hard failure labels on 300k+ drives across 13+ years.
   Requires binning SMART attributes to tokens.

3. **SCANIA Component X** — the only real automotive predictive-maint
   dataset at fleet scale (33,641 trucks, Nature Sci. Data 2025).
   Preserves automotive framing; requires defending the histogram +
   counter → token binning step.

Deprioritised: Google Borg (duplicates the Alibaba domain), N-CMAPSS
(continuous, would raise "you discretised your way to the result").

## Gaps

- SCANIA Nature landing page redirected to auth; schema details from
  PMC mirror + arXiv preprint.
- Bosch Production Line schema not re-verified this session.
- Did not exhaustively enumerate every PHM Society challenge year.
- No 2023-2026 AAAI workshop vehicle-PdM dataset surfaced; absence is
  not proof of non-existence.
