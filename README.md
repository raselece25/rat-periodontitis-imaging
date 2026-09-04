# Multimodal NIR + LSCI Imaging for a Periodontitis Model

Python + MATLAB implementation of a multimodal optical-imaging analysis
pipeline combining Laser Speckle Contrast Imaging (LSCI, blood flow) and
two-wavelength continuous-wave near-infrared spectroscopy (NIR, tissue
oxygenation), plus a Monte Carlo light-transport sanity check linking
the two, and a simple machine-learning classification step over a
synthetic longitudinal animal cohort.

This repository accompanies ongoing, IACUC-approved research at the
**BOIL Lab, Stony Brook University** on micro-endoscopic multimodal
(NIR + LSCI) imaging of LPS-induced periodontitis in a rat model
(IACUC #IACUC2025-00060).

## What this is (and isn't)

This repo implements the **general computational methodology** — LSCI
speckle-contrast-to-blood-flow conversion, modified-Beer-Lambert-law
hemoglobin/oxygenation recovery, a Monte Carlo forward-model sanity
check, and a simple ML classification step — using **entirely synthetic,
randomly generated data**. It does **not** include any unpublished
experimental data, animal-subject measurements, or histological results
from the underlying IACUC-approved study. All 42-animal cohort numbers,
time courses, and classification results here are synthetic
demonstrations, not reported findings. The intent is to give a clear,
runnable, open reference implementation of the underlying signal-
processing and analysis pipeline for portfolio and educational purposes.

## Method overview

1. **LSCI blood-flow processing** (`src/lsci_processing.py`,
   `matlab/lsciContrast.m`): local spatial speckle contrast `K = std/mean`,
   inverted to a correlation time `tau_c` via the single-exposure speckle
   model (Bandyopadhyay et al.; Briers, *J. Biomed. Opt.* 18(6), 066018,
   2013), then to a relative blood-flow index `BFI = 1/tau_c`.
2. **NIR oxygenation** (`src/nir_oxygenation.py`, `matlab/nirHbChanges.m`):
   two-wavelength modified Beer-Lambert law, solved for relative changes
   in oxy-/deoxyhemoglobin concentration and a relative oxygenation
   index. Default extinction coefficients are illustrative literature-
   typical values (see module docstring) — substitute calibrated,
   wavelength-exact coefficients for quantitative use.
3. **Monte Carlo cross-check** (`src/monte_carlo.py`,
   `src/blood_absorption.py`): converts recovered hemoglobin changes into
   a bulk absorption coefficient and runs them through the same
   vectorized single-layer Monte Carlo simulator used in the companion
   [porcine-burn-sfdi](https://github.com/raselece25/porcine-burn-sfdi)
   repository, to sanity-check that increased hemoglobin-driven
   absorption is reflected as reduced NIR diffuse reflectance.
4. **Synthetic longitudinal cohort + ML classification**
   (`src/synthetic_cohort.py`, `src/ml_pipeline.py`): generates a
   synthetic 42-animal (21 LPS / 21 control) longitudinal dataset with
   illustrative inflammation-like trends, then cross-validates a simple
   logistic-regression classifier on peak-response BFI/oxygenation
   features.

## Repository layout

```
rat-periodontitis-imaging/
├── src/
│   ├── lsci_processing.py       # speckle contrast -> tau_c -> BFI
│   ├── nir_oxygenation.py       # two-wavelength MBLL inversion
│   ├── blood_absorption.py      # hemoglobin -> bulk mu_a
│   ├── monte_carlo.py           # vectorized single-layer MC (shared design
│   │                             # with the porcine-burn-sfdi repo)
│   ├── synthetic_cohort.py      # synthetic 42-animal LPS/control dataset
│   └── ml_pipeline.py           # peak-feature extraction + CV classifier
├── matlab/
│   ├── lsciContrast.m
│   ├── nirHbChanges.m
│   └── run_demo.m
├── scripts/run_demo.py          # end-to-end Python demo -> figures/demo_summary.png
├── tests/test_periodontitis.py  # pytest unit tests
├── figures/                     # demo output (git-ignored except .gitkeep)
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
pytest tests/
```

For the MATLAB version, add `matlab/` to your path and run
`matlab/run_demo.m`.

## Citation

If you build on this code, please cite this repository:

```
Ahmmed, R. (2026). Multimodal NIR + LSCI Imaging for a Periodontitis
Model [Software]. https://github.com/raselece25/rat-periodontitis-imaging
```

## License

MIT — see [LICENSE](LICENSE).
