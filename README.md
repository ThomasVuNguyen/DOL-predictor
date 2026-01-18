# DOL-predictor

Predictive modeling pipeline for WHD litigation/penalty risk.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
Place the WHD CSV/JSON files in the repository root. Large source files are ignored by git by default:
- `whd_whisard.csv` (full raw data)
- `whd_predictive_dataset.json` (prepared HF-style dataset)

Generate the JSON dataset from the CSV with:
```bash
python combine_data.py
```

## Training
Run the baseline trainer (multicore histogram gradient boosting):
```bash
python train.py
```

Artifacts (metrics + loss plot) are written to `artifacts/`.
