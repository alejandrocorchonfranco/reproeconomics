# Data

The original datasets used in this repository are not redistributed here.

The empirical analysis combines annual demographic and macroeconomic data for Spain over the period 1976–2022. The demographic component is based on official fertility and population sources, including the Human Fertility Database (HFD) and the Instituto Nacional de Estadística (INE). The macroeconomic component is based on official institutional statistical sources used to construct the annual control variables employed in the econometric analysis.

Users seeking to reproduce the results should obtain the original data directly from the corresponding providers and place the processed input files in this directory.

The code expects the following local input files:

- `BBDD_fertilidad.xlsx`
- `BBDD_macro.csv`

These files are not included in the public repository.

All variables were harmonised to annual frequency and aligned by calendar year. The analytical sample covers Spain, 1976–2022. Missing observations in macroeconomic controls were not imputed and were excluded from specifications requiring those controls.
