#!/usr/bin/env python3
"""
TFG pipeline unificado
======================

Consolidación y depuración de los notebooks del TFG en un único script reproducible.

Qué incluye
-----------
1) Preparación y fusión de las bases de fertilidad y macroeconomía.
2) Interrupted Time Series (ITS) con y sin controles macroeconómicos.
3) Gráficos de ajuste / contrafactual para las variables demográficas principales.
4) Event-study posterior a la intervención.
5) Figuras del análisis económico corregido:
   - mapa bidimensional de sensibilidad del coste por nacimiento adicional
   - tornado de sensibilidad univariante
   - impacto presupuestario
   - utilización por quintiles
   - curvas de concentración

Qué se ha depurado
------------------
- Cargas repetidas del mismo CSV/XLSX con distintos separadores.
- Celdas exploratorias y redefiniciones redundantes.
- Dependencia del estado interno del notebook.
- Valores “fallback” pegados manualmente en celdas intermedias.
- Mezcla de castellano/inglés en nombres de variables y bloques.

Este archivo es una reconstrucción limpia basada sobre todo en:
- TFG-Fifth run.ipynb
- Gráficas del análisis económico.ipynb

Uso
---
python tfg_pipeline_unificado.py \
    --fertility BBDD_fertilidad.xlsx \
    --macro BBDD_macro.csv \
    --outdir outputs_tfg

Dependencias
------------
pip install pandas numpy matplotlib statsmodels scipy openpyxl
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import BoundaryNorm
from scipy.optimize import brentq
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

@dataclass
class Config:
    fertility_path: str
    macro_path: str
    outdir: str
    intervention_year: int = 2010
    hac_maxlags: int = 3


ITS_OUTCOMES = ["adjTFR", "adjTFR1", "adjTFR3plus", "ASFR_35_39", "ASFR_40_44"]
MACRO_CONTROLS = ["GDP_Growth_PC", "Female_LFP", "Juvenile_Unemployment_Rate"]

# Parámetros económicos corregidos recogidos en el notebook final de gráficas
C_U = 5_000.0       # vitrificación + almacenamiento 5 años
C_T = 2_500.0       # ciclo de thaw-and-transfer / IVF
ALPHA = 0.50        # adicionalidad demográfica
NFC = 91_391.0      # contribución fiscal neta por nacimiento adicional
U_UTIL = 0.08       # tasa de utilización base
ADMIN = 2_000_000.0
N_WOMEN = 1_800_000

UPTAKE_SCENARIOS = {
    "low": 0.05,
    "medium": 0.10,
    "high": 0.15,
}


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_current_figure(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _safe_read_wdi_csv(path: str) -> pd.DataFrame:
    """
    Lee BBDD_macro.csv en formato WDI.
    En los notebooks se probaban varios separadores y encabezados;
    aquí se deja una sola versión robusta.
    """
    return pd.read_csv(path, sep=",", header=2, quotechar='"')


# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def load_fertility_data(path: str) -> pd.DataFrame:
    """
    Construye un panel anual con:
    - adjTFR
    - adjTFR1
    - adjTFR3plus
    - ASFR_35_39
    - ASFR_40_44
    """
    # 1) TFR ajustada total
    df_adjTFR = pd.read_excel(path, sheet_name="adjTFR")

    # 2) Paridez / birth order
    df_adjTFR_bo = pd.read_excel(path, sheet_name="ESPadjtfrRRbo")
    df_adjTFR_bo["adjTFR3plus"] = (
        df_adjTFR_bo["adjTFR3"].fillna(0)
        + df_adjTFR_bo["adjTFR4"].fillna(0)
        + df_adjTFR_bo["adjTFR5p"].fillna(0)
    )

    df_tfr = pd.merge(
        df_adjTFR[["Year", "adjTFR"]],
        df_adjTFR_bo[["Year", "adjTFR1", "adjTFR3plus"]],
        on="Year",
        how="left",
    )

    # 3) ASFR y exposiciones
    df_asfr = pd.read_excel(path, sheet_name="ESPasfrRR")
    df_exposure = pd.read_excel(path, sheet_name="ESPexposRR")

    df_asfr = df_asfr.copy()
    df_asfr["Age_numeric"] = pd.to_numeric(df_asfr["Age"], errors="coerce")

    merged = pd.merge(
        df_asfr,
        df_exposure,
        left_on=["Year", "Age_numeric"],
        right_on=["Year", "Age"],
        how="inner",
        suffixes=("_asfr", "_exp"),
    )

    # Intento de detección robusta de nombres de columnas
    asfr_col_candidates = [c for c in merged.columns if "asfr" in c.lower()]
    exposure_col_candidates = [
        c for c in merged.columns
        if ("expo" in c.lower()) or ("expos" in c.lower()) or ("population" in c.lower())
    ]

    if not asfr_col_candidates:
        raise ValueError("No se encontró una columna ASFR en la hoja ESPasfrRR.")
    if not exposure_col_candidates:
        # fallback razonable: primera columna numérica que no sea Year/Age
        exposure_col_candidates = [
            c for c in merged.columns
            if c not in {"Year", "Age", "Age_numeric"} and pd.api.types.is_numeric_dtype(merged[c])
        ]

    asfr_col = asfr_col_candidates[0]
    exposure_col = exposure_col_candidates[0]

    def weighted_band(df: pd.DataFrame, min_age: int, max_age: int, label: str) -> pd.DataFrame:
        band = df[(df["Age_numeric"] >= min_age) & (df["Age_numeric"] <= max_age)].copy()
        grouped = (
            band.groupby("Year")
            .apply(lambda g: np.average(g[asfr_col], weights=g[exposure_col]))
            .reset_index(name=label)
        )
        return grouped

    asfr_35_39 = weighted_band(merged, 35, 39, "ASFR_35_39")
    asfr_40_44 = weighted_band(merged, 40, 44, "ASFR_40_44")

    out = (
        df_tfr.merge(asfr_35_39, on="Year", how="left")
              .merge(asfr_40_44, on="Year", how="left")
              .sort_values("Year")
              .reset_index(drop=True)
    )
    return out


def load_macro_data(path: str) -> pd.DataFrame:
    """
    Extrae series macro para España desde BBDD_macro.csv:
    - GDP_Growth_PC
    - Juvenile_Unemployment_Rate
    - Female_LFP
    """
    df = _safe_read_wdi_csv(path)
    esp = df[df["Country Code"] == "ESP"].copy()
    year_cols = [c for c in esp.columns if c.isdigit()]

    indicator_map = {
        "GDP per capita growth (annual %)": "GDP_Growth_PC",
        "Unemployment, youth total (% of total labor force ages 15-24) (national estimate)": "Juvenile_Unemployment_Rate",
        "Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)": "Female_LFP",
    }

    sub = esp[esp["Indicator Name"].isin(indicator_map.keys())].copy()
    long = sub.melt(
        id_vars=["Indicator Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    long["Indicator"] = long["Indicator Name"].map(indicator_map)

    pivot = (
        long.dropna(subset=["Year", "Value", "Indicator"])
            .pivot_table(index="Year", columns="Indicator", values="Value", aggfunc="first")
            .reset_index()
            .sort_values("Year")
    )
    return pivot


def build_analysis_dataset(fertility_path: str, macro_path: str) -> pd.DataFrame:
    fertility = load_fertility_data(fertility_path)
    macro = load_macro_data(macro_path)

    df = fertility.merge(macro, on="Year", how="left").sort_values("Year").reset_index(drop=True)

    for col in MACRO_CONTROLS:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    return df


# =============================================================================
# ANÁLISIS ITS
# =============================================================================

def add_its_terms(data: pd.DataFrame, intervention_year: int) -> pd.DataFrame:
    df = data.copy()
    df["time"] = df["Year"] - df["Year"].min()
    df["intervention"] = (df["Year"] >= intervention_year).astype(int)
    df["post_intervention_trend"] = df["time"] * df["intervention"]
    return df


def run_its_analysis(
    data: pd.DataFrame,
    outcome_variables: Iterable[str],
    intervention_year: int,
    maxlags: int,
    macro_variables: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, sm.regression.linear_model.RegressionResultsWrapper]]:
    """
    ITS para múltiples outcomes usando OLS con errores HAC (Newey-West).
    """
    macro_variables = macro_variables or []
    df_its = add_its_terms(data, intervention_year)

    base_formula_parts = ["time", "intervention", "post_intervention_trend"]
    predictors = base_formula_parts + macro_variables
    predictors_str = " + ".join(predictors)

    its_results = []
    fitted_models = {}

    for outcome in outcome_variables:
        formula = f"{outcome} ~ {predictors_str}"
        model = ols(formula, data=df_its)
        results = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": maxlags, "use_correction": True},
        )
        fitted_models[outcome] = results

        summary_row = {
            "Outcome": outcome,
            "N": results.nobs,
            "R_squared": results.rsquared,
        }
        for param in results.params.index:
            summary_row[f"{param}_coef"] = results.params[param]
            summary_row[f"{param}_hac_se"] = results.bse[param]
            summary_row[f"{param}_pvalue"] = results.pvalues[param]
            ci = results.conf_int().loc[param]
            summary_row[f"{param}_ci_low"] = ci.iloc[0]
            summary_row[f"{param}_ci_high"] = ci.iloc[1]

        its_results.append(summary_row)

    return pd.DataFrame(its_results), fitted_models


def make_prediction_matrices(
    df: pd.DataFrame,
    macro_variables: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_pred = df.copy()
    cols = ["time", "intervention", "post_intervention_trend"] + macro_variables
    X = sm.add_constant(df_pred[cols], has_constant="add")
    X_cf = X.copy()
    if "intervention" in X_cf.columns:
        X_cf["intervention"] = 0
    if "post_intervention_trend" in X_cf.columns:
        X_cf["post_intervention_trend"] = 0
    return X, X_cf


def compute_event_study(
    fitted_model,
    intervention_year: int,
    last_year: int,
) -> pd.DataFrame:
    beta_level = fitted_model.params["intervention"]
    beta_slope = fitted_model.params["post_intervention_trend"]

    cov = fitted_model.cov_params().loc[
        ["intervention", "post_intervention_trend"],
        ["intervention", "post_intervention_trend"],
    ]
    var_level = cov.loc["intervention", "intervention"]
    var_slope = cov.loc["post_intervention_trend", "post_intervention_trend"]
    covar = cov.loc["intervention", "post_intervention_trend"]

    rows = []
    for year in range(intervention_year, last_year + 1):
        t = year - intervention_year
        effect = beta_level + beta_slope * t
        variance = var_level + (t ** 2) * var_slope + 2 * t * covar
        variance = max(variance, 0.0)
        se = np.sqrt(variance)
        rows.append(
            {
                "Year": year,
                "Effect": effect,
                "Lower_CI": effect - 1.96 * se,
                "Upper_CI": effect + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# GRÁFICOS ITS
# =============================================================================

def plot_its_fit(
    df: pd.DataFrame,
    outcome: str,
    model,
    intervention_year: int,
    macro_variables: List[str],
    output_path: str,
) -> None:
    df_plot = add_its_terms(df, intervention_year)
    X, X_cf = make_prediction_matrices(df_plot, macro_variables)

    df_plot["fitted"] = model.predict(X)
    df_plot["counterfactual"] = model.predict(X_cf)

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot["Year"], df_plot[outcome], marker="o", linewidth=1.8, label="Observed")
    plt.plot(df_plot["Year"], df_plot["fitted"], linewidth=2.0, label="Fitted")
    plt.plot(df_plot["Year"], df_plot["counterfactual"], linestyle="--", linewidth=2.0, label="Counterfactual")
    plt.axvline(intervention_year, color="black", linestyle=":", alpha=0.8)
    plt.title(f"ITS with macro controls: {outcome}")
    plt.xlabel("Year")
    plt.ylabel(outcome)
    plt.legend()
    plt.grid(alpha=0.2)
    save_current_figure(output_path)


def plot_event_study(event_df: pd.DataFrame, intervention_year: int, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.axvline(intervention_year, linestyle=":", linewidth=1.2)
    plt.plot(event_df["Year"], event_df["Effect"], marker="o", linewidth=2)
    plt.fill_between(event_df["Year"], event_df["Lower_CI"], event_df["Upper_CI"], alpha=0.25)
    plt.title("Post-intervention event study (adjTFR)")
    plt.xlabel("Year")
    plt.ylabel("Estimated policy effect")
    plt.grid(alpha=0.2)
    save_current_figure(output_path)


# =============================================================================
# ANÁLISIS ECONÓMICO
# =============================================================================

def cost_per_birth(u, s, alpha=ALPHA, Cu=C_U, Ct=C_T):
    return (Cu + u * Ct) / (u * s * alpha)


def calculate_cost_per_additional_birth(C_vit, u, s, alpha, C_treatment):
    denominator = u * s * alpha
    if denominator == 0:
        return np.inf
    return (C_vit + u * C_treatment) / denominator


def one_way_sensitivity_results() -> Tuple[float, Dict[str, List[Dict[str, float]]]]:
    """
    Reconstrucción depurada de la sensibilidad univariante del notebook final.
    """
    icer_base = calculate_cost_per_additional_birth(
        C_vit=C_U,
        u=0.08,
        s=0.30,
        alpha=0.50,
        C_treatment=C_T,
    )

    ranges = {
        "C_vit_procedure": [4_000.0, 6_000.0],
        "u": [0.05, 0.12],
        "s": [0.20, 0.40],
        "alpha": [0.35, 0.65],
    }

    results = {}
    for param, bounds in ranges.items():
        records = []
        for value in bounds:
            params = {
                "C_vit": C_U,
                "u": 0.08,
                "s": 0.30,
                "alpha": 0.50,
                "C_treatment": C_T,
            }
            if param == "C_vit_procedure":
                params["C_vit"] = value
            elif param == "u":
                params["u"] = value
            elif param == "s":
                params["s"] = value
            elif param == "alpha":
                params["alpha"] = value

            records.append({"Value": value, "ICER": calculate_cost_per_additional_birth(**params)})
        results[param] = records

    return icer_base, results


def compute_annual_budget(
    n_women: int,
    uptake: float,
    c_u: float,
    c_t: float,
    u_util: float,
    admin: float,
) -> Dict[str, float]:
    n_freeze = n_women * uptake
    freeze_cost = n_freeze * c_u
    later_use_cost = n_freeze * u_util * c_t
    total = freeze_cost + later_use_cost + admin
    return {
        "n_freeze": n_freeze,
        "freeze_cost": freeze_cost,
        "later_use_cost": later_use_cost,
        "admin": admin,
        "total": total,
    }


# =============================================================================
# FIGURAS ECONÓMICAS
# =============================================================================

def plot_cost_sensitivity_heatmap(output_path: str) -> None:
    u_vals = np.linspace(0.04, 0.22, 300)
    s_vals = np.linspace(0.15, 0.45, 300)
    UU, SS = np.meshgrid(u_vals, s_vals)
    COST = cost_per_birth(UU, SS)

    boundaries = [0, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000]
    cmap = plt.cm.get_cmap("RdYlGn_r", len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.pcolormesh(u_vals, s_vals, COST, cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(im, ax=ax, ticks=boundaries, pad=0.02)
    cbar.set_label("Cost per additional live birth (€)", fontsize=10)
    cbar.ax.set_yticklabels(
        [f"€{v/1e3:.0f}k" if v < 1e6 else f"€{v/1e6:.1f}M" for v in boundaries]
    )

    cs = ax.contour(u_vals, s_vals, COST, levels=[NFC], colors=["navy"], linewidths=2.0, linestyles="--")
    ax.clabel(cs, fmt=f"Break-even (≈ €{NFC/1e3:.0f}k)", fontsize=8, inline=True)

    u_base, s_base = 0.08, 0.30
    base_cost = cost_per_birth(u_base, s_base)
    ax.plot(u_base, s_base, marker="^", markersize=9, label=f"Base case: €{base_cost:,.0f}")

    ax.axvline(x=0.05, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.axvline(x=0.12, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.fill_betweenx(
        [s_vals.min(), s_vals.max()],
        0.05,
        0.12,
        alpha=0.07,
        label="Empirical utilisation range (5–12%)",
    )

    ax.set_xlabel("Utilisation rate u")
    ax.set_ylabel("Success rate s")
    ax.set_title("Cost per additional live birth")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tornado(output_path: str) -> None:
    icer_base, sensitivity = one_way_sensitivity_results()

    rows = []
    for param, results in sensitivity.items():
        icer_low = results[0]["ICER"]
        icer_high = results[1]["ICER"]
        rows.append(
            {
                "Parameter": param,
                "Deviation_Low": icer_low - icer_base,
                "Deviation_High": icer_high - icer_base,
                "Absolute_Impact": abs(icer_high - icer_low),
            }
        )

    df = pd.DataFrame(rows).sort_values("Absolute_Impact", ascending=True).reset_index(drop=True)
    names = {
        "C_vit_procedure": "Coste de vitrificación",
        "u": "Tasa de utilización",
        "s": "Tasa de éxito",
        "alpha": "Adicionalidad",
    }
    df["Label"] = df["Parameter"].map(names)

    plt.figure(figsize=(10, 6))
    y = np.arange(len(df))
    plt.barh(y, df["Deviation_Low"], height=0.6, label="Lower bound")
    plt.barh(y, df["Deviation_High"], height=0.6, label="Upper bound")
    plt.yticks(y, df["Label"])
    plt.axvline(0, linestyle="--", linewidth=1.2)
    plt.xlabel("Deviation from base-case ICER (€)")
    plt.title("One-way sensitivity analysis (tornado)")
    plt.legend()
    plt.grid(axis="x", alpha=0.2)
    save_current_figure(output_path)


def load_macro_spain_for_distribution(path: str) -> pd.DataFrame:
    df = _safe_read_wdi_csv(path)
    esp = df[df["Country Code"] == "ESP"].copy()
    year_cols = [c for c in esp.columns if c.isdigit()]
    target = [
        "Gini index",
        "Income share held by lowest 20%",
        "Income share held by second 20%",
        "Income share held by third 20%",
        "Income share held by fourth 20%",
        "Income share held by highest 20%",
    ]
    sub = esp[esp["Indicator Name"].isin(target)].copy()
    long = sub.melt(
        id_vars=["Indicator Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    return long.dropna(subset=["Year", "Value"])


def extract_latest_quintile_shares(macro_long: pd.DataFrame) -> Dict[str, float]:
    labels = {
        "Gini index": "Gini",
        "Income share held by lowest 20%": "Q1",
        "Income share held by second 20%": "Q2",
        "Income share held by third 20%": "Q3",
        "Income share held by fourth 20%": "Q4",
        "Income share held by highest 20%": "Q5",
    }

    out = {}
    for indicator, short in labels.items():
        sub = macro_long[macro_long["Indicator Name"] == indicator].sort_values("Year")
        last = sub.dropna(subset=["Value"]).iloc[-1]
        out[short] = float(last["Value"])
        out[f"{short}_year"] = int(last["Year"])
    return out


def concentration_index(rates: np.ndarray, income_shares: np.ndarray) -> float:
    """
    Concentration index discreto sobre quintiles.
    """
    weights = income_shares / income_shares.sum()
    cum_pop_mid = np.cumsum(np.repeat(0.2, len(rates))) - 0.1
    mu = np.sum(weights * rates)
    if mu == 0:
        return 0.0
    return 2 * np.sum(weights * rates * cum_pop_mid) / mu - 1


def calibrate_universal_profile(avg_programme: float, target_ci: float, income_shares: np.ndarray) -> np.ndarray:
    """
    Calibra un perfil lineal por quintiles con media fijada y CI objetivo.
    """
    ranks = np.arange(1, 6)

    def profile_from_b(b):
        raw = 1 + b * (ranks - ranks.mean())
        raw = np.maximum(raw, 1e-6)
        weights = np.ones_like(raw) / len(raw)
        scale = avg_programme / np.sum(weights * raw)
        return raw * scale

    def objective(b):
        prof = profile_from_b(b)
        return concentration_index(prof, income_shares) - target_ci

    b_star = brentq(objective, -0.9, 0.9)
    return profile_from_b(b_star)


def calibrate_utilisation_profiles(
    income_shares: np.ndarray,
    q5_q1_ratio: float = 14.0,
    ci_universal: float = 0.18,
    ci_meanstested: float = 0.05,
    avg_private: float = 0.03,
    avg_programme: float = 0.10,
) -> Dict[str, np.ndarray]:
    # Perfil privado con razón Q5/Q1 fijada
    q1 = avg_private / ((1 + 1 + 1 + 1 + q5_q1_ratio) / 5)
    private = np.array([q1, q1, q1, q1, q1 * q5_q1_ratio])

    universal = calibrate_universal_profile(avg_programme, ci_universal, income_shares)
    means_tested = calibrate_universal_profile(avg_programme, ci_meanstested, income_shares)

    return {
        "private": private,
        "universal": universal,
        "means_tested": means_tested,
        "ci_private": concentration_index(private, income_shares),
        "ci_universal": concentration_index(universal, income_shares),
        "ci_mt": concentration_index(means_tested, income_shares),
    }


def plot_budget_impact(output_path: str) -> None:
    rows = []
    for label, uptake in UPTAKE_SCENARIOS.items():
        vals = compute_annual_budget(N_WOMEN, uptake, C_U, C_T, U_UTIL, ADMIN)
        vals["scenario"] = label
        vals["uptake"] = uptake
        rows.append(vals)

    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 6))
    plt.bar(df["scenario"], df["freeze_cost"] / 1e6, label="Freeze + storage")
    plt.bar(
        df["scenario"],
        df["later_use_cost"] / 1e6,
        bottom=df["freeze_cost"] / 1e6,
        label="Later treatment use",
    )
    plt.bar(
        df["scenario"],
        df["admin"] / 1e6,
        bottom=(df["freeze_cost"] + df["later_use_cost"]) / 1e6,
        label="Administration",
    )
    plt.ylabel("Annual budget (€ millions)")
    plt.title("Annual budget impact under uptake scenarios")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    save_current_figure(output_path)


def plot_quintiles(profiles: Dict[str, np.ndarray], gini_data: Dict[str, float], output_path: str) -> None:
    quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(quintiles))
    width = 0.24

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, profiles["private"] * 100, width=width, label="Private")
    plt.bar(x, profiles["universal"] * 100, width=width, label="Universal")
    plt.bar(x + width, profiles["means_tested"] * 100, width=width, label="Means-tested")
    plt.xticks(x, quintiles)
    plt.ylabel("Estimated utilisation rate (%)")
    plt.title(f"Estimated oocyte utilisation by income quintile (Gini {gini_data['Gini']:.1f})")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    save_current_figure(output_path)


def plot_concentration_curves(profiles: Dict[str, np.ndarray], output_path: str) -> None:
    x = np.linspace(0.2, 1.0, 5)

    def curve(vals):
        vals = np.array(vals, dtype=float)
        cum = np.cumsum(vals) / np.sum(vals)
        return np.insert(cum, 0, 0.0)

    x_full = np.insert(x, 0, 0.0)

    plt.figure(figsize=(8, 8))
    plt.plot(x_full, x_full, linestyle="--", linewidth=1.5, label="Equality line")
    plt.plot(x_full, curve(profiles["private"]), linewidth=2, label="Private")
    plt.plot(x_full, curve(profiles["universal"]), linewidth=2, label="Universal")
    plt.plot(x_full, curve(profiles["means_tested"]), linewidth=2, label="Means-tested")
    plt.xlabel("Cumulative population share")
    plt.ylabel("Cumulative utilisation share")
    plt.title("Concentration curves")
    plt.legend()
    plt.grid(alpha=0.2)
    save_current_figure(output_path)


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline unificado del TFG")
    parser.add_argument("--fertility", required=True, help="Ruta a BBDD_fertilidad.xlsx")
    parser.add_argument("--macro", required=True, help="Ruta a BBDD_macro.csv")
    parser.add_argument("--outdir", default="outputs_tfg", help="Directorio de salida")
    parser.add_argument("--intervention-year", type=int, default=2010)
    parser.add_argument("--hac-maxlags", type=int, default=3)
    args = parser.parse_args()

    cfg = Config(
        fertility_path=args.fertility,
        macro_path=args.macro,
        outdir=args.outdir,
        intervention_year=args.intervention_year,
        hac_maxlags=args.hac_maxlags,
    )
    ensure_outdir(cfg.outdir)

    # 1) Dataset analítico
    df = build_analysis_dataset(cfg.fertility_path, cfg.macro_path)

    # 2) ITS principal
    summary_macro, models_macro = run_its_analysis(
        data=df,
        outcome_variables=ITS_OUTCOMES,
        intervention_year=cfg.intervention_year,
        maxlags=cfg.hac_maxlags,
        macro_variables=MACRO_CONTROLS,
    )

    summary_no_macro, models_no_macro = run_its_analysis(
        data=df,
        outcome_variables=["adjTFR"],
        intervention_year=cfg.intervention_year,
        maxlags=cfg.hac_maxlags,
        macro_variables=[],
    )

    summary_macro.to_csv(os.path.join(cfg.outdir, "its_summary_with_macro.csv"), index=False)
    summary_no_macro.to_csv(os.path.join(cfg.outdir, "its_summary_adjTFR_no_macro.csv"), index=False)

    # 3) Gráficos ITS
    for outcome in ITS_OUTCOMES:
        plot_its_fit(
            df=df,
            outcome=outcome,
            model=models_macro[outcome],
            intervention_year=cfg.intervention_year,
            macro_variables=MACRO_CONTROLS,
            output_path=os.path.join(cfg.outdir, f"its_fit_{outcome}.png"),
        )

    event_df = compute_event_study(
        fitted_model=models_macro["adjTFR"],
        intervention_year=cfg.intervention_year,
        last_year=int(df["Year"].max()),
    )
    event_df.to_csv(os.path.join(cfg.outdir, "event_study_adjTFR.csv"), index=False)
    plot_event_study(
        event_df=event_df,
        intervention_year=cfg.intervention_year,
        output_path=os.path.join(cfg.outdir, "event_study_adjTFR.png"),
    )

    # 4) Figuras económicas
    plot_cost_sensitivity_heatmap(os.path.join(cfg.outdir, "fig_cost_sensitivity.png"))
    plot_tornado(os.path.join(cfg.outdir, "fig_tornado.png"))
    plot_budget_impact(os.path.join(cfg.outdir, "fig_budget_impact.png"))

    macro_long = load_macro_spain_for_distribution(cfg.macro_path)
    gini_data = extract_latest_quintile_shares(macro_long)
    income_shares = np.array([gini_data["Q1"], gini_data["Q2"], gini_data["Q3"], gini_data["Q4"], gini_data["Q5"]])
    profiles = calibrate_utilisation_profiles(income_shares=income_shares)

    plot_quintiles(profiles, gini_data, os.path.join(cfg.outdir, "fig_quintiles.png"))
    plot_concentration_curves(profiles, os.path.join(cfg.outdir, "fig_concentration.png"))

    # 5) Resumen textual
    with open(os.path.join(cfg.outdir, "README_salida.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "Pipeline unificado ejecutado correctamente.\n\n"
            "Ficheros generados:\n"
            "- its_summary_with_macro.csv\n"
            "- its_summary_adjTFR_no_macro.csv\n"
            "- event_study_adjTFR.csv\n"
            "- its_fit_*.png\n"
            "- event_study_adjTFR.png\n"
            "- fig_cost_sensitivity.png\n"
            "- fig_tornado.png\n"
            "- fig_budget_impact.png\n"
            "- fig_quintiles.png\n"
            "- fig_concentration.png\n"
        )

    print(f"Proceso completado. Salida en: {os.path.abspath(cfg.outdir)}")


if __name__ == "__main__":
    main()
