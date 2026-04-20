# Airbnb Madrid — What drives prices?

Reproducible pipeline for InsideAirbnb Madrid listings: ingestion, cleaning, feature engineering, modeling (OLS/Ridge/Lasso/XGBoost/LightGBM), interpretation, and report figures.

## How to reproduce

### Environment
Python 3.11+ required.

```bash
pip install -r requirements.txt
```

### Run
Open `notebooks/main_analysis.ipynb` and run all cells top to bottom.
Data will be downloaded automatically on first run.
Expected total runtime: ~10–20 minutes on a modern laptop (dominated by CV).

### Data
Primary data: InsideAirbnb Madrid (auto-downloaded, no account needed).
Metro stations: Madrid Open Data portal (auto-downloaded, no account needed).

### Outputs
All figures saved to `outputs/figures/`.
All model metrics saved to `outputs/model_results.csv`.

## Notes

- The Ayuntamiento metro GeoJSON URL in the project occasionally returns 404 when the portal reorganizes datasets. In that case, `src/ingest.py` falls back to OpenStreetMap (Overpass) station coordinates inside the Madrid metro area.
