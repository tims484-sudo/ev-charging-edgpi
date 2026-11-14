# EV Charging Infrastructure Analysis - India (EDGPI)

Code for the dissertation *Data-Driven Analysis and Visualisation of Electric Vehicle
Charging Infrastructure in India*.

## What it does
- RQ1-RQ3: temporal patterns, geographic gaps (Gini), dashboard summary tables/charts
- RQ4: **EDGPI** (Equity-weighted Demand-Gap Priority Index) + marginal-gain station
  allocation, validated against random, proportional, guideline and demand-only baselines
- RQ4b: EDGPI compared with three existing methods, re-implemented from their published steps
  - M1 CRITIC-weighted deficit index (Maric & Ravnjak, 2026, doi:10.1111/tgis.70261)
  - M2 Entropy-weight TOPSIS (Hwang & Yoon, 1981)
  - M3 Equity-first allocation (supply-per-EV parity logic of Cai et al., 2024; Huang et al., 2026)

## Run
```
pip install -r requirements.txt
# put the five CSV files in ./data/ (names are listed at the top of EV.py)
python EV.py
```
Outputs (CSV + PNG) are written to `./outputs/`. RQ4b files are `rq4b_*`.
The random seed is fixed (`RNG_SEED = 42`).

## Data sources
VAHAN registrations, OpenChargeMap, PlugShare, BEE FAME-II utilisation, Census 2011 projections.
Raw files are not committed; see the dissertation (Chapter 3.4) for access details.
