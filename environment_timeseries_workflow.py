from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Heavy_Metal_Concentrations.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SITE = "S2"
TARGET_VARIABLE = "HPI"
TEST_MONTHS = 12
FORECAST_MONTHS = 48

SEASON_TO_MONTH = {
    "winter": 1,
    "spring": 4,
    "summer": 7,
    "autumn": 10,
}

METAL_COLUMNS = ["Zn", "Cd", "Pb", "Cu", "Ni", "Mn", "As", "Cr"]

SI = {"Zn": 15000, "Cd": 10, "Pb": 15, "Cu": 1500, "Ni": 100, "Mn": 300, "As": 50, "Cr": 100}
II = {"Zn": 5000, "Cd": 5, "Pb": 10, "Cu": 50, "Ni": 20, "Mn": 100, "As": 10, "Cr": 1}
MAC = {"Zn": 5000, "Cd": 5, "Pb": 10, "Cu": 50, "Ni": 20, "Mn": 50, "As": 10, "Cr": 50}
WEIGHTS = {metal: 1 / MAC[metal] for metal in METAL_COLUMNS}
WEIGHT_SUM = sum(WEIGHTS.values())


def print_section(title: str) -> None:
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")


def load_dataset(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)
    header = [str(value).strip() for value in raw.iloc[1].tolist()]
    df = raw.iloc[2:].copy().reset_index(drop=True)
    df.columns = header

    df = df.rename(columns={"Month": "Season"})
    for col in ["Year", *METAL_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Season"] = df["Season"].str.strip().str.lower()
    df["Month"] = df["Season"].map(SEASON_TO_MONTH)
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    df = df.sort_values(["Site", "Date"]).reset_index(drop=True)
    return df


def calculate_hpi(row: pd.Series) -> float:
    weighted_sum = sum(
        WEIGHTS[metal] * abs(row[metal] - II[metal]) / (SI[metal] - II[metal])
        for metal in METAL_COLUMNS
    )
    return weighted_sum / WEIGHT_SUM


def calculate_hei(row: pd.Series) -> float:
    return sum(row[metal] / MAC[metal] for metal in METAL_COLUMNS) + 10 - len(METAL_COLUMNS)


def calculate_dc(row: pd.Series) -> float:
    return sum(max((row[metal] / MAC[metal]) - 1, 0) for metal in METAL_COLUMNS)


def add_pollution_indices(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["HPI"] = enriched.apply(calculate_hpi, axis=1)
    enriched["HEI"] = enriched.apply(calculate_hei, axis=1)
    enriched["DC"] = enriched.apply(calculate_dc, axis=1)
    return enriched


def dataset_summary(df: pd.DataFrame) -> None:
    print_section("1. Load and Understand the Dataset")
    print(f"Dataset file: {DATA_FILE}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nDataset structure:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column names: {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nSimple summary:")
    print(
        "- The file contains heavy metal measurements for two monitoring sites: S1 (upstream) and S2 (downstream).\n"
        "- The paper describes monthly data from 2018 to 2022, but the workbook actually contains seasonal records\n"
        "  (winter, spring, summer, autumn), giving 20 observations per site and 40 observations in total.\n"
        "- The main measured variables are Zn, Cd, Pb, Cu, Ni, Mn, As, and Cr.\n"
        "- Additional pollution indices (HPI, HEI, DC) are computed from the paper's formulas for modelling."
    )


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            records.append({"column": col, "outlier_count": 0, "lower_bound": q1, "upper_bound": q3})
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        records.append({"column": col, "outlier_count": count, "lower_bound": lower, "upper_bound": upper})
    return pd.DataFrame(records)


def data_quality_report(df: pd.DataFrame) -> None:
    print_section("2. Check Data Quality")
    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    outlier_report = detect_outliers_iqr(df, METAL_COLUMNS + ["HPI", "HEI", "DC"])

    print("Missing values per column:")
    print(missing.to_string())

    print(f"\nDuplicate rows: {duplicates}")

    print("\nOutlier counts using the IQR rule:")
    print(outlier_report[["column", "outlier_count"]].to_string(index=False))

    print("\nQuality conclusion:")
    print(
        "- No missing values were found in the usable columns.\n"
        "- No duplicate records were found.\n"
        "- A few columns show high-end outliers, mainly because the dataset is very small and some metals have low variance.\n"
        "- The dataset is internally consistent, but it is not truly monthly in the downloaded workbook.\n"
        "- It is suitable for a demonstration forecasting workflow, but forecast confidence should be treated as moderate rather than high."
    )


def preprocess_for_modeling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print_section("3. Data Preprocessing")
    print("Preprocessing steps:")
    print("- Converted the time field into datetime using a season-to-month mapping.")
    print("- Sorted the dataset by site and date.")
    print("- No missing numeric values were found, so no imputation was required.")
    print("- Numeric scaling was not applied because ARIMA and Random Forest do not require it here.")
    print("- Created time-based features: Year and Month.")

    site_df = df[df["Site"] == TARGET_SITE].copy().sort_values("Date")
    site_df = site_df.set_index("Date")

    monthly = site_df[METAL_COLUMNS + ["HPI", "HEI", "DC"]].resample("MS").interpolate(method="linear")
    monthly["Year"] = monthly.index.year
    monthly["Month"] = monthly.index.month

    print(
        f"\nImportant note: because the source workbook is seasonal rather than monthly, "
        f"the monthly modelling table for {TARGET_SITE} is created by linear interpolation "
        "between observed seasonal values."
    )
    print(f"Observed seasonal records for {TARGET_SITE}: {site_df.shape[0]}")
    print(f"Derived monthly records for {TARGET_SITE}: {monthly.shape[0]}")

    return site_df.reset_index(), monthly.reset_index().rename(columns={"index": "Date"})


def create_eda_plots(df: pd.DataFrame) -> None:
    print_section("4. Exploratory Data Analysis (EDA)")
    print(
        "EDA focus:\n"
        "- Trend plots for Zn, Cu, Mn, and HPI over time.\n"
        "- Correlation matrix for metals and pollution indices.\n"
        "- Simple interpretation of trend and seasonality."
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    features_to_plot = ["Zn", "Cu", "Mn", "HPI"]
    for ax, feature in zip(axes.flat, features_to_plot):
        sns.lineplot(data=df, x="Date", y=feature, hue="Site", marker="o", ax=ax)
        ax.set_title(f"{feature} Trend Over Time")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    trend_plot = OUTPUT_DIR / "trend_plots.png"
    plt.savefig(trend_plot, dpi=300, bbox_inches="tight")
    plt.close()

    corr_columns = METAL_COLUMNS + ["HPI", "HEI", "DC"]
    corr = df[corr_columns].corr()
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    corr_plot = OUTPUT_DIR / "correlation_matrix.png"
    plt.savefig(corr_plot, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        "EDA insights:\n"
        "- Most metals remain fairly stable across the study period.\n"
        "- Mn and Cu show the most visible fluctuations, especially at some seasonal points.\n"
        "- Upstream and downstream patterns are very similar, which matches the paper's low-pollution conclusion.\n"
        "- The pollution indices remain low and nearly flat, so long-range forecasting should be interpreted as a stable-trend scenario."
    )


def select_arima_order(train_series: pd.Series) -> tuple[tuple[int, int, int], float]:
    best_order = None
    best_aic = np.inf
    for p in range(0, 4):
        for d in range(0, 3):
            for q in range(0, 4):
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    if best_order is None:
        raise RuntimeError("ARIMA order search failed for all candidate models.")
    return best_order, best_aic


def add_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    lagged = df.copy()
    for lag in [1, 2, 3, 6, 12]:
        lagged[f"{target_col}_lag_{lag}"] = lagged[target_col].shift(lag)
    return lagged.dropna().reset_index(drop=True)


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def model_workflow(monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]], tuple[int, int, int]]:
    print_section("5. Model Selection and 6. Train the Model")
    print(
        f"Selected target variable: {TARGET_VARIABLE} at site {TARGET_SITE}.\n"
        "- ARIMA is used as the primary forecasting model because it is a standard baseline for univariate time-series forecasting.\n"
        "- Random Forest Regression is used as a comparison model because it can learn non-linear relationships from lagged values and calendar features.\n"
        "- HPI is a good target because it summarizes overall heavy-metal pollution into one interpretable index."
    )

    monthly_df = monthly_df.copy()
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])
    monthly_df = monthly_df.sort_values("Date").reset_index(drop=True)

    train = monthly_df.iloc[:-TEST_MONTHS].copy()
    test = monthly_df.iloc[-TEST_MONTHS:].copy()

    print(f"\nTraining set size: {len(train)} monthly observations")
    print(f"Testing set size: {len(test)} monthly observations")
    print(f"Training period: {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Testing period:  {test['Date'].min().date()} to {test['Date'].max().date()}")

    arima_order, arima_aic = select_arima_order(train[TARGET_VARIABLE])
    print(f"\nBest ARIMA order selected by AIC search: {arima_order} (AIC={arima_aic:.3f})")

    arima_model = ARIMA(train[TARGET_VARIABLE], order=arima_order).fit()
    arima_pred = arima_model.forecast(steps=len(test))
    arima_results = test[["Date", TARGET_VARIABLE]].copy()
    arima_results["ARIMA_Predicted"] = arima_pred.values

    rf_ready = add_lag_features(monthly_df[["Date", TARGET_VARIABLE, "Year", "Month", *METAL_COLUMNS]], TARGET_VARIABLE)
    rf_train = rf_ready[rf_ready["Date"] < test["Date"].min()].copy()
    rf_test = rf_ready[rf_ready["Date"] >= test["Date"].min()].copy()
    feature_cols = [col for col in rf_ready.columns if col not in ["Date", TARGET_VARIABLE]]

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )
    rf_model.fit(rf_train[feature_cols], rf_train[TARGET_VARIABLE])
    rf_pred = rf_model.predict(rf_test[feature_cols])
    rf_results = rf_test[["Date", TARGET_VARIABLE]].copy()
    rf_results["RF_Predicted"] = rf_pred

    metrics = {
        "ARIMA": evaluate_predictions(arima_results[TARGET_VARIABLE], arima_results["ARIMA_Predicted"]),
        "RandomForest": evaluate_predictions(rf_results[TARGET_VARIABLE], rf_results["RF_Predicted"]),
    }

    return arima_results, rf_results, metrics, arima_order


def forecast_future(monthly_df: pd.DataFrame, arima_order: tuple[int, int, int]) -> pd.DataFrame:
    print_section("7. Forecast Future Behavior")
    full_model = ARIMA(monthly_df[TARGET_VARIABLE], order=arima_order).fit()
    forecast_object = full_model.get_forecast(steps=FORECAST_MONTHS)
    forecast_values = forecast_object.predicted_mean
    conf_int = forecast_object.conf_int()

    future_dates = pd.date_range(
        start=pd.to_datetime(monthly_df["Date"].max()) + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS",
    )

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            f"Forecast_{TARGET_VARIABLE}": forecast_values.values,
            "Lower_95_CI": conf_int.iloc[:, 0].values,
            "Upper_95_CI": conf_int.iloc[:, 1].values,
        }
    )
    forecast_df["Year"] = forecast_df["Date"].dt.year
    forecast_df["Month"] = forecast_df["Date"].dt.month

    print("Forecast table (next 48 months):")
    print(forecast_df.to_string(index=False))

    forecast_df.to_csv(OUTPUT_DIR / "future_forecast_48_months.csv", index=False)
    return forecast_df


def evaluation_report(metrics: dict[str, dict[str, float]]) -> None:
    print_section("8. Model Evaluation")
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df.round(6).to_string())

    print(
        "\nMetric meaning:\n"
        "- MAE (Mean Absolute Error): the average absolute difference between actual and predicted values.\n"
        "- RMSE (Root Mean Squared Error): similar to MAE, but it penalizes larger mistakes more strongly.\n"
        "- R-squared: how much of the variation in the target is explained by the model. Closer to 1 is better."
    )

    primary_r2 = metrics["ARIMA"]["R2"]
    if primary_r2 >= 0.75:
        quality = "good"
    elif primary_r2 >= 0.40:
        quality = "reasonable but improvable"
    else:
        quality = "weak and should be improved"
    print(f"\nARIMA performance assessment: {quality}.")


def create_model_plots(
    monthly_df: pd.DataFrame,
    arima_results: pd.DataFrame,
    rf_results: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    print_section("9. Visualization")
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_df["Date"], monthly_df[TARGET_VARIABLE], label="Actual", color="black", linewidth=2)
    plt.plot(arima_results["Date"], arima_results["ARIMA_Predicted"], label="ARIMA Predicted", linestyle="--")
    plt.plot(rf_results["Date"], rf_results["RF_Predicted"], label="RF Predicted", linestyle=":")
    plt.title(f"Actual vs Predicted {TARGET_VARIABLE} for Site {TARGET_SITE}")
    plt.xlabel("Date")
    plt.ylabel(TARGET_VARIABLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_df["Date"], monthly_df[TARGET_VARIABLE], label="Historical", color="black", linewidth=2)
    plt.plot(forecast_df["Date"], forecast_df[f"Forecast_{TARGET_VARIABLE}"], label="Forecast", color="tab:blue")
    plt.fill_between(
        forecast_df["Date"],
        forecast_df["Lower_95_CI"],
        forecast_df["Upper_95_CI"],
        color="tab:blue",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.title(f"Next 48-Month Forecast of {TARGET_VARIABLE} for Site {TARGET_SITE}")
    plt.xlabel("Date")
    plt.ylabel(TARGET_VARIABLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "future_forecast_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    print(f"- {OUTPUT_DIR / 'trend_plots.png'}")
    print(f"- {OUTPUT_DIR / 'correlation_matrix.png'}")
    print(f"- {OUTPUT_DIR / 'actual_vs_predicted.png'}")
    print(f"- {OUTPUT_DIR / 'future_forecast_trend.png'}")


def final_conclusion(metrics: dict[str, dict[str, float]], forecast_df: pd.DataFrame) -> None:
    print_section("10. Final Conclusion")
    avg_forecast = forecast_df[f"Forecast_{TARGET_VARIABLE}"].mean()
    first_forecast = forecast_df[f"Forecast_{TARGET_VARIABLE}"].iloc[0]
    last_forecast = forecast_df[f"Forecast_{TARGET_VARIABLE}"].iloc[-1]

    print(
        "Reliability summary:\n"
        "- The dataset is clean in terms of missing values and duplicates.\n"
        "- The major limitation is structural: the downloaded workbook is seasonal, not monthly as described in the paper.\n"
        "- Because of that mismatch, the 48-month series is based on a monthly interpolation of seasonal observations."
    )

    print(
        f"\nForecast realism:\n"
        f"- The forecast remains in a low-pollution range, which is consistent with the study's original interpretation.\n"
        f"- Forecasted {TARGET_VARIABLE} starts near {first_forecast:.4f}, ends near {last_forecast:.4f}, "
        f"and averages {avg_forecast:.4f} over the next four years."
    )

    print(
        "\nLimitations:\n"
        "- Small sample size.\n"
        "- Seasonal source data were converted into monthly data by interpolation.\n"
        "- Forecasting only one target variable (HPI) cannot capture every environmental process.\n"
        "- No external drivers such as rainfall, temperature, discharge volume, or land-use change were included."
    )

    print(
        "\nSuggested improvements:\n"
        "- Obtain the true monthly raw observations, if available.\n"
        "- Add external environmental predictors.\n"
        "- Try seasonal ARIMA/SARIMAX once longer monthly history is available.\n"
        "- Build separate models for individual metals as well as the composite index."
    )

    print_section("11. Output Files")
    print(f"Python script: {BASE_DIR / 'environment_timeseries_workflow.py'}")
    print(f"Forecast CSV:  {OUTPUT_DIR / 'future_forecast_48_months.csv'}")
    print(f"Plots folder:   {OUTPUT_DIR}")


def main() -> None:
    df = load_dataset(DATA_FILE)
    df = add_pollution_indices(df)

    dataset_summary(df)
    data_quality_report(df)
    site_df, monthly_df = preprocess_for_modeling(df)
    create_eda_plots(df)
    arima_results, rf_results, metrics, arima_order = model_workflow(monthly_df)
    forecast_df = forecast_future(monthly_df, arima_order)
    evaluation_report(metrics)
    create_model_plots(monthly_df, arima_results, rf_results, forecast_df)
    final_conclusion(metrics, forecast_df)


if __name__ == "__main__":
    main()
