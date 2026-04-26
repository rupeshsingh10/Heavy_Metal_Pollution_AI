# Heavy_Metal_Pollution_AI

This repository contains the workflow and generated outputs for a time-series analysis of heavy metal concentration data.

## Contents

- `environment_timeseries_workflow.py`: end-to-end analysis and forecasting script
- `Heavy_Metal_Concentrations.xlsx`: source dataset used by the workflow
- `Pollution_Indices.xls`: supporting spreadsheet file
- `outputs/`: generated plots and forecast CSV
- `environment_timeseries_outputs.zip`: bundled archive of the main script and generated outputs

## Generated Outputs

The `outputs/` folder includes:

- `actual_vs_predicted.png`
- `correlation_matrix.png`
- `future_forecast_48_months.csv`
- `future_forecast_trend.png`
- `trend_plots.png`

## Notes

- The workflow targets site `S2` and forecasts the `HPI` variable.
- The script derives pollution indices and generates both evaluation plots and a 48-month forecast.
- The repository includes both the extracted output files and the zip archive for convenience.
