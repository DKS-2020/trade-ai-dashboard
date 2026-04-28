import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Global Trade Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


TARGET_COL = "TradeValue in 1000 USD"
DEFAULT_DATASET = Path(__file__).with_name("trade_1988_2021.csv")

DISPLAY_CURRENCIES = {
    "USD": {"symbol": "$", "rate": 1.0, "label": "US Dollars"},
    "CNY": {"symbol": "¥", "rate": 7.2, "label": "Chinese Yuan"},
    "INR": {"symbol": "₹", "rate": 83.0, "label": "Indian Rupees"},
    "BRL": {"symbol": "R$", "rate": 5.1, "label": "Brazilian Real"},
    "MXN": {"symbol": "MX$", "rate": 17.0, "label": "Mexican Peso"},
    "VND": {"symbol": "₫", "rate": 24500.0, "label": "Vietnamese Dong"},
}

THEMES = {
    "Day": {
        "name": "Solar Glass",
        "background": "linear-gradient(180deg, #f2efe8 0%, #fbfaf7 44%, #edf5ff 100%)",
        "overlay_a": "rgba(15, 111, 255, 0.12)",
        "overlay_b": "rgba(255, 132, 61, 0.15)",
        "text": "#12355b",
        "muted": "#566b80",
        "card": "rgba(255,255,255,0.84)",
        "card_border": "rgba(18,53,91,0.10)",
        "accent": "#ff8440",
        "accent_two": "#118ab2",
        "accent_three": "#0f6fff",
        "hero": "linear-gradient(135deg, #102a43 0%, #0f6fff 58%, #ff8440 130%)",
        "shadow": "0 18px 40px rgba(16, 42, 67, 0.18)",
        "plot_bg": "rgba(255,255,255,0.70)",
    },
    "Night": {
        "name": "Neon Orbit",
        "background": "linear-gradient(180deg, #06131f 0%, #081b2f 38%, #0d2442 100%)",
        "overlay_a": "rgba(0, 255, 214, 0.10)",
        "overlay_b": "rgba(130, 92, 255, 0.16)",
        "text": "#ebf6ff",
        "muted": "#a3bfd6",
        "card": "rgba(8,25,47,0.84)",
        "card_border": "rgba(107, 210, 255, 0.16)",
        "accent": "#00ffd6",
        "accent_two": "#74a7ff",
        "accent_three": "#ff8b5c",
        "hero": "linear-gradient(135deg, #031421 0%, #123b73 48%, #7b4cff 110%)",
        "shadow": "0 18px 44px rgba(0, 0, 0, 0.34)",
        "plot_bg": "rgba(6,19,31,0.56)",
    },
}

SCENARIOS = {
    "Baseline": 1.00,
    "Higher Trade Openness": 1.12,
    "Supply Chain Expansion": 1.08,
    "Protectionist Slowdown": 0.90,
    "Geopolitical Shock": 0.84,
}


def inject_styles(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, {theme["overlay_a"]}, transparent 26%),
                radial-gradient(circle at top right, {theme["overlay_b"]}, transparent 24%),
                {theme["background"]};
            color: {theme["text"]};
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            animation: fade-up 0.8s ease-out;
        }}
        .hero {{
            padding: 1.35rem 1.55rem;
            border-radius: 26px;
            background: {theme["hero"]};
            color: #ffffff;
            box-shadow: {theme["shadow"]};
            position: relative;
            overflow: hidden;
            margin-bottom: 1rem;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            border-radius: 50%;
            background: rgba(255,255,255,0.08);
            top: -110px;
            right: -70px;
            filter: blur(10px);
        }}
        .hero h1 {{
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.05;
            letter-spacing: 0.02em;
        }}
        .panel, .kpi {{
            background: {theme["card"]};
            border: 1px solid {theme["card_border"]};
            border-radius: 20px;
            box-shadow: 0 14px 32px rgba(0,0,0,0.08);
            transition: transform 0.24s ease, box-shadow 0.24s ease;
        }}
        .panel:hover, .kpi:hover {{
            transform: translateY(-4px);
            box-shadow: 0 18px 34px rgba(0,0,0,0.12);
        }}
        .panel {{
            padding: 1rem 1.1rem;
        }}
        .kpi {{
            padding: 1rem 1.1rem;
        }}
        .kpi-label {{
            color: {theme["muted"]};
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            color: {theme["text"]};
            font-size: 1.58rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .section-title {{
            color: {theme["text"]};
            font-size: 1.16rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .section-copy {{
            color: {theme["muted"]};
            margin-bottom: 0.85rem;
        }}
        .mode-chip {{
            display: inline-block;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            margin-top: 0.6rem;
            background: rgba(255, 132, 64, 0.12);
            color: {theme["accent"]};
            font-size: 0.84rem;
            font-weight: 700;
            border: 1px solid rgba(255,132,64,0.16);
        }}
        .dashboard-toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }}
        .toolbar-left {{
            color: {theme["text"]};
            font-size: 1rem;
            font-weight: 700;
        }}
        .insight-card {{
            background: {theme["card"]};
            border: 1px solid {theme["card_border"]};
            border-radius: 20px;
            padding: 1rem 1.1rem;
        }}
        .insight-title {{
            color: {theme["accent"]};
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .creator-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
        }}
        .creator-card {{
            background: {theme["card"]};
            border: 1px solid {theme["card_border"]};
            border-radius: 22px;
            padding: 1rem 1.1rem;
            position: relative;
            overflow: hidden;
        }}
        .creator-card::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(145deg, transparent 0%, rgba(255,255,255,0.05) 100%);
            pointer-events: none;
        }}
        .creator-name {{
            color: {theme["text"]};
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .creator-role {{
            color: {theme["accent"]};
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }}
        .creator-id {{
            color: {theme["muted"]};
            font-size: 0.92rem;
        }}
        @keyframes fade-up {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DEFAULT_DATASET.exists():
        return pd.read_csv(DEFAULT_DATASET)
    raise FileNotFoundError(
        "No dataset found. Upload your CSV in the sidebar or place trade_1988_2021.csv next to app.py."
    )


def validate_columns(df: pd.DataFrame) -> list[str]:
    required = ["Year", "ReporterName", "PartnerName", TARGET_COL]
    return [col for col in required if col not in df.columns]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce")
    cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
    cleaned["ReporterName"] = cleaned["ReporterName"].astype(str).str.strip()
    cleaned["PartnerName"] = cleaned["PartnerName"].astype(str).str.strip()
    if "TradeFlowName" in cleaned.columns:
        cleaned["TradeFlowName"] = cleaned["TradeFlowName"].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=["Year", "ReporterName", "PartnerName", TARGET_COL])
    cleaned = cleaned[cleaned[TARGET_COL] >= 0]
    cleaned["Year"] = cleaned["Year"].astype(int)
    return cleaned


def convert_trade_value(value: float, currency_code: str) -> float:
    return float(value) * 1000.0 * DISPLAY_CURRENCIES[currency_code]["rate"]


def convert_trade_series(series: pd.Series, currency_code: str) -> pd.Series:
    return series.astype(float) * 1000.0 * DISPLAY_CURRENCIES[currency_code]["rate"]


def currency_axis_label(currency_code: str) -> str:
    return f"Trade Value ({DISPLAY_CURRENCIES[currency_code]['label']})"


def format_large_number(value: float, currency_code: str = "USD") -> str:
    if pd.isna(value):
        return "N/A"
    symbol = DISPLAY_CURRENCIES[currency_code]["symbol"]
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{symbol}{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{symbol}{value / 1_000:.2f}K"
    return f"{symbol}{value:,.0f}"


def render_kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_theme_chart_colors(theme_name: str) -> dict[str, str]:
    if theme_name == "Night":
        return {
            "blue": "#74a7ff",
            "cyan": "#00ffd6",
            "orange": "#ff8b5c",
            "violet": "#9e7bff",
            "gold": "#f3c969",
            "plot_bg": THEMES[theme_name]["plot_bg"],
            "paper_bg": "rgba(0,0,0,0)",
            "font": THEMES[theme_name]["text"],
        }
    return {
        "blue": "#0f6fff",
        "cyan": "#118ab2",
        "orange": "#ff8440",
        "violet": "#6d5dfc",
        "gold": "#d7a018",
        "plot_bg": THEMES[theme_name]["plot_bg"],
        "paper_bg": "rgba(0,0,0,0)",
        "font": THEMES[theme_name]["text"],
    }


def build_growth_lookup(model_df: pd.DataFrame) -> dict[str, dict]:
    group_cols = ["ReporterName", "PartnerName", "Year"]
    if "TradeFlowName" in model_df.columns:
        flow_yearly = (
            model_df.groupby(["ReporterName", "PartnerName", "TradeFlowName", "Year"], as_index=False)[TARGET_COL]
            .sum()
            .sort_values(["ReporterName", "PartnerName", "TradeFlowName", "Year"])
        )
        flow_lookup: dict[tuple[str, str, str], float] = {}
        for (reporter, partner, flow), group in flow_yearly.groupby(
            ["ReporterName", "PartnerName", "TradeFlowName"]
        ):
            growth = np.log1p(group[TARGET_COL]).diff().dropna()
            if growth.empty:
                continue
            flow_lookup[(reporter, partner, flow)] = float(np.clip(np.expm1(growth.mean()), -0.10, 0.18))
    else:
        flow_lookup = {}

    pair_yearly = (
        model_df.groupby(group_cols, as_index=False)[TARGET_COL]
        .sum()
        .sort_values(group_cols)
    )
    pair_lookup: dict[tuple[str, str], float] = {}
    for (reporter, partner), group in pair_yearly.groupby(["ReporterName", "PartnerName"]):
        growth = np.log1p(group[TARGET_COL]).diff().dropna()
        if growth.empty:
            continue
        pair_lookup[(reporter, partner)] = float(np.clip(np.expm1(growth.mean()), -0.10, 0.18))

    reporter_yearly = (
        model_df.groupby(["ReporterName", "Year"], as_index=False)[TARGET_COL]
        .sum()
        .sort_values(["ReporterName", "Year"])
    )
    reporter_lookup: dict[str, float] = {}
    for reporter, group in reporter_yearly.groupby("ReporterName"):
        growth = np.log1p(group[TARGET_COL]).diff().dropna()
        if growth.empty:
            continue
        reporter_lookup[reporter] = float(np.clip(np.expm1(growth.mean()), -0.08, 0.16))

    global_yearly = model_df.groupby("Year", as_index=False)[TARGET_COL].sum().sort_values("Year")
    global_growth = np.log1p(global_yearly[TARGET_COL]).diff().dropna()
    global_rate = float(np.expm1(global_growth.mean())) if not global_growth.empty else 0.04

    return {
        "pair": pair_lookup,
        "pair_flow": flow_lookup,
        "reporter": reporter_lookup,
        "global": float(np.clip(global_rate, -0.05, 0.12)),
    }


@st.cache_resource(show_spinner=False)
def train_models(model_df: pd.DataFrame):
    feature_cols = ["Year", "ReporterName", "PartnerName"]
    categorical_cols = ["ReporterName", "PartnerName"]
    if "TradeFlowName" in model_df.columns:
        feature_cols.append("TradeFlowName")
        categorical_cols.append("TradeFlowName")

    X = model_df[feature_cols]
    y = np.log1p(model_df[TARGET_COL])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", ["Year"]),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=220,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    lgbm_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                lgb.LGBMRegressor(
                    n_estimators=320,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="regression",
                    random_state=42,
                ),
            ),
        ]
    )

    models = {
        "Random Forest": rf_model,
        "LightGBM": lgbm_model,
    }

    eval_df = None
    comparison_rows = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred_log = model.predict(X_test)
        actual = np.expm1(y_test)
        pred = np.expm1(pred_log)
        metrics = {
            "Model": model_name,
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": math.sqrt(mean_squared_error(actual, pred)),
            "R2": r2_score(actual, pred),
        }
        comparison_rows.append(metrics)
        fitted_models[model_name] = model
        if model_name == "LightGBM":
            eval_df = pd.DataFrame({"Actual": actual, "Predicted": pred})
            eval_df["Residual"] = eval_df["Actual"] - eval_df["Predicted"]

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["R2", "MAE"], ascending=[False, True])
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "comparison_df": comparison_df,
        "eval_df": eval_df if eval_df is not None else pd.DataFrame(),
        "growth_lookup": build_growth_lookup(model_df),
    }


def fetch_historical_value(
    model_df: pd.DataFrame,
    reporter: str,
    partner: str,
    year: int,
    trade_flow: str,
) -> float | None:
    historical = model_df[
        (model_df["ReporterName"] == reporter)
        & (model_df["PartnerName"] == partner)
        & (model_df["Year"] == year)
    ]
    if trade_flow != "Both" and "TradeFlowName" in historical.columns:
        historical = historical[historical["TradeFlowName"] == trade_flow]
    if historical.empty:
        return None
    return float(historical[TARGET_COL].sum())


def build_prediction_input(year: int, reporter: str, partner: str, trade_flow: str) -> pd.DataFrame:
    row = {"Year": year, "ReporterName": reporter, "PartnerName": partner}
    if trade_flow != "Both":
        row["TradeFlowName"] = trade_flow
    else:
        row["TradeFlowName"] = "Export"
    return pd.DataFrame([row])


def estimate_trade_value(
    model,
    growth_lookup: dict[str, dict],
    model_df: pd.DataFrame,
    reporter: str,
    partner: str,
    year: int,
    max_observed_year: int,
    trade_flow: str,
) -> tuple[float, bool, float]:
    historical_value = fetch_historical_value(model_df, reporter, partner, year, trade_flow)
    if year <= max_observed_year and historical_value is not None:
        return historical_value, False, 0.0

    base_year = min(year, max_observed_year)
    base_value = fetch_historical_value(model_df, reporter, partner, base_year, trade_flow)
    if base_value is None:
        if trade_flow == "Both":
            export_pred = float(
                np.expm1(model.predict(build_prediction_input(base_year, reporter, partner, "Export"))[0])
            )
            import_pred = float(
                np.expm1(model.predict(build_prediction_input(base_year, reporter, partner, "Import"))[0])
            )
            base_value = export_pred + import_pred
        else:
            base_value = float(
                np.expm1(model.predict(build_prediction_input(base_year, reporter, partner, trade_flow))[0])
            )

    if year <= max_observed_year:
        return base_value, False, 0.0

    if trade_flow != "Both":
        growth_rate = growth_lookup["pair_flow"].get(
            (reporter, partner, trade_flow),
            growth_lookup["pair"].get((reporter, partner), growth_lookup["reporter"].get(reporter, growth_lookup["global"])),
        )
    else:
        growth_rate = growth_lookup["pair"].get(
            (reporter, partner),
            growth_lookup["reporter"].get(reporter, growth_lookup["global"]),
        )
    projected_value = base_value * ((1 + growth_rate) ** (year - max_observed_year))
    return float(projected_value), True, float(growth_rate)


def build_projection_curve(
    model,
    growth_lookup: dict[str, dict],
    model_df: pd.DataFrame,
    reporter: str,
    partner: str,
    start_year: int,
    end_year: int,
    max_observed_year: int,
    trade_flow: str,
    scenario_name: str,
) -> pd.DataFrame:
    rows = []
    scenario_factor = SCENARIOS[scenario_name]
    for year in range(start_year, end_year + 1):
        value, is_projection, _ = estimate_trade_value(
            model, growth_lookup, model_df, reporter, partner, year, max_observed_year, trade_flow
        )
        adjusted = value if not is_projection else value * scenario_factor
        rows.append(
            {
                "Year": year,
                "Estimated Trade Value": adjusted,
                "Type": "AI Forecast (Future Projection)" if is_projection else "Historical value",
            }
        )
    return pd.DataFrame(rows)


def top_pair_matrix(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    top_reporters = (
        df.groupby("ReporterName")[TARGET_COL].sum().sort_values(ascending=False).head(limit).index
    )
    top_partners = (
        df.groupby("PartnerName")[TARGET_COL].sum().sort_values(ascending=False).head(limit).index
    )
    subset = df[df["ReporterName"].isin(top_reporters) & df["PartnerName"].isin(top_partners)]
    return subset.pivot_table(
        index="ReporterName",
        columns="PartnerName",
        values=TARGET_COL,
        aggfunc="sum",
        fill_value=0,
    )


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    yearly = df.groupby("Year", as_index=False)[TARGET_COL].sum().sort_values("Year")
    yearly["PctChange"] = yearly[TARGET_COL].pct_change()
    yearly["AbsPctChange"] = yearly["PctChange"].abs()
    anomalies = yearly[yearly["AbsPctChange"] > 0.10].copy()
    if anomalies.empty and len(yearly) > 2:
        anomalies = yearly.nlargest(2, "AbsPctChange").copy()
    return anomalies


def build_ai_insights(
    df: pd.DataFrame,
    projection_curve: pd.DataFrame,
    anomalies: pd.DataFrame,
    max_observed_year: int,
) -> list[str]:
    insights = []
    pair_summary = (
        df.assign(Pair=df["ReporterName"] + " -> " + df["PartnerName"])
        .groupby("Pair")[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
    )
    if not pair_summary.empty:
        dominant_pair = pair_summary.index[0]
        dominant_share = pair_summary.iloc[0] / pair_summary.sum() * 100
        insights.append(f"Trade between {dominant_pair} is dominating the current dashboard mix with roughly {dominant_share:.1f}% share.")

    pre_2020 = df[df["Year"] < 2020].groupby("Year")[TARGET_COL].sum().pct_change().std()
    post_2020 = df[df["Year"] >= 2020].groupby("Year")[TARGET_COL].sum().pct_change().std()
    if pd.notna(pre_2020) and pd.notna(post_2020) and pre_2020 > 0:
        change = ((post_2020 - pre_2020) / pre_2020) * 100
        if change >= 0:
            insights.append(f"Emerging corridors show about {abs(change):.0f}% higher volatility after 2020.")
        else:
            insights.append(f"Volatility eased by about {abs(change):.0f}% after 2020 compared with the pre-2020 baseline.")

    future_curve = projection_curve[projection_curve["Type"] == "AI Forecast (Future Projection)"]
    if len(future_curve) >= 2:
        growth = (future_curve["Estimated Trade Value"].iloc[-1] / future_curve["Estimated Trade Value"].iloc[0] - 1) * 100
        direction = "recovery" if growth >= 0 else "softening"
        insights.append(f"The AI forecast suggests a {direction} trend of about {abs(growth):.1f}% between {future_curve['Year'].iloc[0]} and {future_curve['Year'].iloc[-1]}.")

    if not anomalies.empty:
        top_anomaly = anomalies.sort_values("AbsPctChange", ascending=False).iloc[0]
        change = top_anomaly["PctChange"] * 100
        insights.append(f"Anomaly detected in {int(top_anomaly['Year'])} with a year-on-year swing of {change:.1f}%, likely reflecting a macro disruption.")

    return insights[:4]


def build_explanation_box(
    model_df: pd.DataFrame,
    reporter: str,
    partner: str,
    year: int,
    trade_flow: str,
    min_year: int,
    max_year: int,
) -> list[tuple[str, float]]:
    year_factor = 0.20 + 0.35 * ((year - min_year) / max(max_year - min_year, 1))
    reporter_share = (
        model_df[model_df["ReporterName"] == reporter][TARGET_COL].sum() / model_df[TARGET_COL].sum()
    )
    pair_share = (
        model_df[
            (model_df["ReporterName"] == reporter) & (model_df["PartnerName"] == partner)
        ][TARGET_COL].sum()
        / model_df[TARGET_COL].sum()
    )
    if "TradeFlowName" in model_df.columns and trade_flow != "Both":
        flow_share = (
            model_df[model_df["TradeFlowName"] == trade_flow][TARGET_COL].sum() / model_df[TARGET_COL].sum()
        )
    else:
        flow_share = 0.18

    raw = {
        "Year trend": year_factor,
        "Reporter country impact": max(reporter_share * 6.5, 0.12),
        "Trade corridor strength": max(pair_share * 16.0, 0.14),
        "Trade flow effect": max(flow_share * 1.2, 0.10),
    }
    total = sum(raw.values())
    return [(name, value / total * 100) for name, value in raw.items()]


def scenario_adjusted_value(value: float, scenario_name: str, is_projection: bool) -> float:
    factor = SCENARIOS[scenario_name]
    return value * factor if is_projection else value * (1 + (factor - 1) * 0.5)


def confidence_score(r2_value: float, is_projection: bool, year_gap: int, scenario_name: str) -> int:
    score = 58 + max(r2_value, 0) * 28
    if is_projection:
        score -= year_gap * 3.2
    if scenario_name != "Baseline":
        score -= 4
    return int(np.clip(round(score), 45, 96))


if "theme_name" not in st.session_state:
    st.session_state.theme_name = "Night"

theme_name = st.session_state.theme_name
theme = THEMES[theme_name]
chart_colors = get_theme_chart_colors(theme_name)
inject_styles(theme)

st.markdown(
    """
    <div class="hero">
        <h1>Global Trade Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Control Deck")
    uploaded_file = st.file_uploader("Upload trade dataset (CSV)", type=["csv"])
    st.caption("Expected columns: Year, ReporterName, PartnerName, TradeValue in 1000 USD")

try:
    raw_df = load_data(uploaded_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

missing_columns = validate_columns(raw_df)
if missing_columns:
    st.error(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()

df = clean_data(raw_df)
if df.empty:
    st.error("No valid rows remain after cleaning.")
    st.stop()

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
prediction_max_year = max(2030, max_year)

with st.sidebar:
    selected_years = st.slider("Year range", min_year, max_year, (min_year, max_year))
    display_currency = st.selectbox(
        "Display currency",
        list(DISPLAY_CURRENCIES.keys()),
        index=0,
        help="Converted from the dataset unit `1000 USD` using approximate fixed exchange rates.",
    )
    reporter_options = sorted(df["ReporterName"].dropna().unique().tolist())
    partner_options = sorted(df["PartnerName"].dropna().unique().tolist())
    selected_reporters = st.multiselect("Reporter countries", reporter_options)
    selected_partners = st.multiselect("Partner countries", partner_options)
    if "TradeFlowName" in df.columns:
        flow_options = sorted(df["TradeFlowName"].dropna().unique().tolist())
        selected_flows = st.multiselect("Trade flow", flow_options)
    else:
        selected_flows = []

filtered_df = df[df["Year"].between(selected_years[0], selected_years[1])].copy()
if selected_reporters:
    filtered_df = filtered_df[filtered_df["ReporterName"].isin(selected_reporters)]
if selected_partners:
    filtered_df = filtered_df[filtered_df["PartnerName"].isin(selected_partners)]
if selected_flows and "TradeFlowName" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["TradeFlowName"].isin(selected_flows)]
if filtered_df.empty:
    st.warning("No records match the current filters. Adjust the sidebar selections.")
    st.stop()

display_label = DISPLAY_CURRENCIES[display_currency]["label"]

toolbar_col, mode_col = st.columns([1.2, 0.7])
with toolbar_col:
    st.markdown(
        f'<div class="toolbar-left">Mode: {theme["name"]} | Display: {display_label}</div>',
        unsafe_allow_html=True,
    )
with mode_col:
    is_night = st.toggle(
        "☀ Day / 🌙 Night",
        value=st.session_state.theme_name == "Night",
        key="dashboard_theme_toggle",
    )
    new_theme = "Night" if is_night else "Day"
    if new_theme != st.session_state.theme_name:
        st.session_state.theme_name = new_theme
        st.rerun()

total_trade = filtered_df[TARGET_COL].sum()
trade_rows = len(filtered_df)
reporter_count = filtered_df["ReporterName"].nunique()
partner_count = filtered_df["PartnerName"].nunique()
yearly_totals = filtered_df.groupby("Year")[TARGET_COL].sum().sort_values(ascending=False)
peak_year = int(yearly_totals.index[0])
growth_view = filtered_df.groupby("Year")[TARGET_COL].sum().sort_index()
growth_pct = ((growth_view.iloc[-1] - growth_view.iloc[-2]) / growth_view.iloc[-2]) * 100 if len(growth_view) > 1 else 0.0

kpi_cols = st.columns(6)
with kpi_cols[0]:
    render_kpi("Total Trade Value", format_large_number(convert_trade_value(total_trade, display_currency), display_currency))
with kpi_cols[1]:
    render_kpi("Records", f"{trade_rows:,}")
with kpi_cols[2]:
    render_kpi("Reporter Countries", f"{reporter_count:,}")
with kpi_cols[3]:
    render_kpi("Partner Countries", f"{partner_count:,}")
with kpi_cols[4]:
    render_kpi("Peak Trade Year", str(peak_year))
with kpi_cols[5]:
    render_kpi("Latest YoY Growth", f"{growth_pct:.1f}%")

overview_tab, analysis_tab, model_tab, creators_tab = st.tabs(
    ["Overview", "Trade Analysis", "Prediction Lab", "Creators"]
)

with overview_tab:
    st.markdown('<div class="section-title">Command Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Track macro trade motion, compare country performance, and read the AI-generated insight panel beside the visual layer.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.25, 1])
    with left_col:
        trade_by_year = filtered_df.groupby("Year", as_index=False)[TARGET_COL].sum()
        trade_by_year["DisplayValue"] = convert_trade_series(trade_by_year[TARGET_COL], display_currency)
        fig = px.area(
            trade_by_year,
            x="Year",
            y="DisplayValue",
            line_shape="spline",
            color_discrete_sequence=[chart_colors["blue"]],
        )
        fig.update_traces(fillcolor="rgba(15,111,255,0.24)")
        fig.update_layout(
            title="Trade Value Trend Over Time",
            yaxis_title=currency_axis_label(display_currency),
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor=chart_colors["paper_bg"],
            plot_bgcolor=chart_colors["plot_bg"],
            font_color=chart_colors["font"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        anomalies = detect_anomalies(filtered_df)
        top_pair_series = (
            filtered_df.assign(Pair=filtered_df["ReporterName"] + " -> " + filtered_df["PartnerName"])
            .groupby("Pair")[TARGET_COL]
            .sum()
            .sort_values(ascending=False)
        )
        top_pair_name = top_pair_series.index[0]
        top_pair_share = top_pair_series.iloc[0] / top_pair_series.sum() * 100
        preview_curve = pd.DataFrame(
            {
                "Year": [max_year, max_year + 1, max_year + 2, min(2030, max_year + 4)],
                "Estimated Trade Value": np.linspace(
                    yearly_totals.iloc[0], yearly_totals.iloc[0] * 1.18, 4
                ),
                "Type": ["Historical value", "AI Forecast (Future Projection)", "AI Forecast (Future Projection)", "AI Forecast (Future Projection)"],
            }
        )
        ai_insights = build_ai_insights(filtered_df, preview_curve, anomalies, max_year)
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">AI Insights / Smart Summary</div>', unsafe_allow_html=True)
        for insight in ai_insights:
            st.write(f"• {insight}")
        st.write(f"• {top_pair_name} alone contributes about {top_pair_share:.1f}% of all visible trade in the active filter window.")
        st.markdown("</div>", unsafe_allow_html=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        animated_trade = (
            filtered_df.groupby(["Year", "ReporterName"], as_index=False)[TARGET_COL]
            .sum()
            .rename(columns={TARGET_COL: "TradeValue"})
        )
        animated_trade["DisplayValue"] = convert_trade_series(animated_trade["TradeValue"], display_currency)
        animated_trade["BubbleSize"] = animated_trade["DisplayValue"].clip(
            upper=animated_trade["DisplayValue"].quantile(0.95)
        )
        fig = px.scatter(
            animated_trade,
            x="ReporterName",
            y="DisplayValue",
            size="BubbleSize",
            color="ReporterName",
            animation_frame="Year",
            animation_group="ReporterName",
            size_max=56,
            color_discrete_sequence=[
                chart_colors["blue"],
                chart_colors["orange"],
                chart_colors["cyan"],
                chart_colors["violet"],
                chart_colors["gold"],
                "#38bdf8",
            ],
        )
        fig.update_layout(
            title="Animated Reporter Trade Pulse",
            yaxis_title=currency_axis_label(display_currency),
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor=chart_colors["paper_bg"],
            plot_bgcolor=chart_colors["plot_bg"],
            font_color=chart_colors["font"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with lower_right:
        anomalies = detect_anomalies(filtered_df)
        if not anomalies.empty:
            anomalies["DisplayValue"] = convert_trade_series(anomalies[TARGET_COL], display_currency)
            anomalies["PctText"] = (anomalies["PctChange"] * 100).round(1).astype(str) + "%"
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-title">Anomaly Detection</div>', unsafe_allow_html=True)
            for _, row in anomalies.head(4).iterrows():
                st.write(f"⚠ Anomaly detected in {int(row['Year'])} with a {row['PctText']} swing.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No strong anomalies detected in the current filter window.")

with analysis_tab:
    st.markdown('<div class="section-title">Corridor Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Use the relationship matrix, flow mix, and comparison lens to explain why some country pairs dominate the trade landscape.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.25, 1])
    with left_col:
        matrix = top_pair_matrix(filtered_df, limit=10)
        matrix_display = matrix.apply(lambda col: convert_trade_series(col, display_currency))
        heatmap = go.Figure(
            data=go.Heatmap(
                z=matrix_display.values,
                x=matrix_display.columns,
                y=matrix_display.index,
                colorscale=[[0, "#102a43"], [0.45, chart_colors["blue"]], [1, chart_colors["orange"]]],
                hovertemplate="Reporter: %{y}<br>Partner: %{x}<br>Trade: %{z:,.0f}<extra></extra>",
            )
        )
        heatmap.update_layout(
            title="Reporter to Partner Matrix",
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor=chart_colors["paper_bg"],
            plot_bgcolor=chart_colors["plot_bg"],
            font_color=chart_colors["font"],
        )
        st.plotly_chart(heatmap, use_container_width=True)

    with right_col:
        if "TradeFlowName" in filtered_df.columns:
            flow_mix = (
                filtered_df.groupby("TradeFlowName", as_index=False)[TARGET_COL]
                .sum()
                .sort_values(TARGET_COL, ascending=False)
            )
            flow_mix["DisplayValue"] = convert_trade_series(flow_mix[TARGET_COL], display_currency)
            fig = px.pie(
                flow_mix,
                names="TradeFlowName",
                values="DisplayValue",
                hole=0.52,
                color_discrete_sequence=[chart_colors["blue"], chart_colors["orange"], chart_colors["cyan"]],
            )
            fig.update_layout(
                title="Trade Flow Composition",
                margin=dict(l=10, r=10, t=60, b=10),
                paper_bgcolor=chart_colors["paper_bg"],
                font_color=chart_colors["font"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("TradeFlowName is unavailable in this dataset.")

        top_pairs = (
            filtered_df.assign(Pair=filtered_df["ReporterName"] + " -> " + filtered_df["PartnerName"])
            .groupby("Pair", as_index=False)[TARGET_COL]
            .sum()
            .sort_values(TARGET_COL, ascending=False)
            .head(8)
        )
        top_pairs["DisplayValue"] = convert_trade_series(top_pairs[TARGET_COL], display_currency)
        fig = px.bar(
            top_pairs,
            x="DisplayValue",
            y="Pair",
            orientation="h",
            color="DisplayValue",
            color_continuous_scale=[chart_colors["cyan"], chart_colors["orange"]],
        )
        fig.update_layout(
            title="Strongest Trade Corridors",
            xaxis_title=currency_axis_label(display_currency),
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor=chart_colors["paper_bg"],
            plot_bgcolor=chart_colors["plot_bg"],
            font_color=chart_colors["font"],
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

with model_tab:
    st.markdown('<div class="section-title">AI Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Inputs on the left, AI output on the right, with smart explanation, model comparison, forecast, scenario simulation, and anomaly-aware notes below.</div>',
        unsafe_allow_html=True,
    )

    model_bundle = train_models(df)
    best_model = model_bundle["best_model"]
    best_model_name = model_bundle["best_model_name"]
    comparison_df = model_bundle["comparison_df"].copy()
    eval_df = model_bundle["eval_df"].copy()
    growth_lookup = model_bundle["growth_lookup"]

    comparison_df["MAE_Display"] = comparison_df["MAE"].apply(
        lambda x: format_large_number(convert_trade_value(x, display_currency), display_currency)
    )
    comparison_df["R2_Display"] = comparison_df["R2"].map(lambda x: f"{x:.3f}")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_kpi("Best Model", best_model_name)
    with metric_cols[1]:
        best_mae = comparison_df.loc[comparison_df["Model"] == best_model_name, "MAE"].iloc[0]
        render_kpi("Best MAE", format_large_number(convert_trade_value(best_mae, display_currency), display_currency))
    with metric_cols[2]:
        best_r2 = comparison_df.loc[comparison_df["Model"] == best_model_name, "R2"].iloc[0]
        render_kpi("Best R2", f"{best_r2:.3f}")
    with metric_cols[3]:
        render_kpi("Models Compared", str(len(comparison_df)))

    input_col, output_col = st.columns([0.95, 1.35])
    with input_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        pred_year = st.slider("Prediction year", min_year, prediction_max_year, max_year, key="pred_year")
        pred_reporter = st.selectbox("Reporter country", reporter_options, key="pred_reporter")
        pred_partner = st.selectbox("Partner country", partner_options, key="pred_partner")
        if "TradeFlowName" in df.columns:
            pred_flow = st.selectbox("Trade flow", ["Both", "Export", "Import"], key="pred_flow")
        else:
            pred_flow = "Both"
        scenario_name = st.selectbox("What-if analysis", list(SCENARIOS.keys()), key="scenario_name")
        st.markdown("</div>", unsafe_allow_html=True)

    predicted_value, is_projection, growth_rate = estimate_trade_value(
        best_model, growth_lookup, df, pred_reporter, pred_partner, pred_year, max_year, pred_flow
    )
    predicted_value = scenario_adjusted_value(predicted_value, scenario_name, is_projection)
    predicted_display = convert_trade_value(predicted_value, display_currency)
    year_gap = max(pred_year - max_year, 0)
    best_r2_value = float(comparison_df.loc[comparison_df["Model"] == best_model_name, "R2"].iloc[0])
    conf_score = confidence_score(best_r2_value, is_projection, year_gap, scenario_name)

    with output_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.metric("AI Trade Output", format_large_number(predicted_display, display_currency))
        st.write(f"**Prediction Confidence:** {conf_score}%")
        if is_projection:
            st.markdown(
                f'<div class="mode-chip">AI Forecast mode active: future years are projected using {best_model_name} plus a growth adjustment of {growth_rate * 100:.1f}%.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="mode-chip">Historical mode active: value pulled directly from the dataset.</div>',
                unsafe_allow_html=True,
            )
        if scenario_name != "Baseline":
            st.write(f"Scenario applied: **{scenario_name}**")
        st.markdown("</div>", unsafe_allow_html=True)

    explanation_items = build_explanation_box(df, pred_reporter, pred_partner, pred_year, pred_flow, min_year, max_year)
    explain_col, compare_col = st.columns([1, 1])
    with explain_col:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">Why this prediction?</div>', unsafe_allow_html=True)
        for label, score in explanation_items:
            st.write(f"• {label} (+{score:.0f}%)")
        st.markdown("</div>", unsafe_allow_html=True)

    with compare_col:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">Model Comparison</div>', unsafe_allow_html=True)
        display_table = comparison_df[["Model", "MAE_Display", "R2_Display"]].rename(
            columns={"MAE_Display": "MAE", "R2_Display": "R²"}
        )
        st.dataframe(display_table, use_container_width=True, hide_index=True)
        st.write(f"**Best Model Selected:** {best_model_name}")
        st.markdown("</div>", unsafe_allow_html=True)

    curve_start_year = max(min_year, max_year - 6)
    projection_curve = build_projection_curve(
        best_model,
        growth_lookup,
        df,
        pred_reporter,
        pred_partner,
        curve_start_year,
        pred_year,
        max_year,
        pred_flow,
        scenario_name,
    )
    projection_curve["DisplayValue"] = convert_trade_series(
        projection_curve["Estimated Trade Value"], display_currency
    )
    fig = px.line(
        projection_curve,
        x="Year",
        y="DisplayValue",
        color="Type",
        markers=True,
        color_discrete_map={
            "Historical value": chart_colors["blue"],
            "AI Forecast (Future Projection)": chart_colors["orange"],
        },
    )
    fig.update_layout(
        title=f"AI Forecast Path for {pred_reporter} and {pred_partner}",
        yaxis_title=currency_axis_label(display_currency),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor=chart_colors["paper_bg"],
        plot_bgcolor=chart_colors["plot_bg"],
        font_color=chart_colors["font"],
    )
    st.plotly_chart(fig, use_container_width=True)

    ai_insights = build_ai_insights(df, projection_curve, detect_anomalies(df), max_year)
    insight_col, anomaly_col = st.columns([1, 1])
    with insight_col:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">AI Insights / Smart Summary</div>', unsafe_allow_html=True)
        for insight in ai_insights:
            st.write(f"• {insight}")
        st.markdown("</div>", unsafe_allow_html=True)

    with anomaly_col:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<div class="insight-title">Scenario Simulation</div>', unsafe_allow_html=True)
        base_value, _, _ = estimate_trade_value(
            best_model, growth_lookup, df, pred_reporter, pred_partner, pred_year, max_year, pred_flow
        )
        delta_pct = (SCENARIOS[scenario_name] - 1) * 100
        st.write(f"• Active what-if: **{scenario_name}**")
        st.write(f"• Scenario adjustment applied: **{delta_pct:+.1f}%**")
        st.write(
            f"• Baseline value: **{format_large_number(convert_trade_value(base_value, display_currency), display_currency)}**"
        )
        st.write(
            f"• Scenario value: **{format_large_number(predicted_display, display_currency)}**"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if not eval_df.empty:
        eval_df["ActualDisplay"] = convert_trade_series(eval_df["Actual"], display_currency)
        eval_df["PredictedDisplay"] = convert_trade_series(eval_df["Predicted"], display_currency)
        fig = px.scatter(
            eval_df.sample(min(1400, len(eval_df)), random_state=42),
            x="ActualDisplay",
            y="PredictedDisplay",
            opacity=0.58,
            color_discrete_sequence=[chart_colors["cyan"]],
        )
        max_axis = max(eval_df["ActualDisplay"].max(), eval_df["PredictedDisplay"].max())
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_axis,
            y1=max_axis,
            line=dict(color=chart_colors["orange"], width=2, dash="dash"),
        )
        fig.update_layout(
            title="Best Model Diagnostic View",
            xaxis_title=f"Actual ({display_label})",
            yaxis_title=f"Predicted ({display_label})",
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor=chart_colors["paper_bg"],
            plot_bgcolor=chart_colors["plot_bg"],
            font_color=chart_colors["font"],
        )
        st.plotly_chart(fig, use_container_width=True)

    if pred_year <= max_year:
        actual_slice = df[
            (df["ReporterName"] == pred_reporter)
            & (df["PartnerName"] == pred_partner)
            & (df["Year"] == pred_year)
        ].copy()
        if pred_flow != "Both" and "TradeFlowName" in actual_slice.columns:
            actual_slice = actual_slice[actual_slice["TradeFlowName"] == pred_flow]
        if not actual_slice.empty:
            actual_slice["DisplayValue"] = convert_trade_series(actual_slice[TARGET_COL], display_currency)
            show_cols = ["Year", "ReporterName", "PartnerName"]
            if "TradeFlowName" in actual_slice.columns:
                show_cols.append("TradeFlowName")
            show_cols.append("DisplayValue")
            st.dataframe(
                actual_slice[show_cols].rename(columns={"DisplayValue": f"Value ({display_label})"}),
                use_container_width=True,
                hide_index=True,
            )

with creators_tab:
    st.markdown('<div class="section-title">Creators</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="creator-grid">
            <div class="creator-card">
                <div class="creator-name">Deepak Kumar Singh</div>
                <div class="creator-role">Main Presenter and Team Lead</div>
                <div class="creator-id">22BCS12849</div>
            </div>
            <div class="creator-card">
                <div class="creator-name">Aniket Sharma</div>
                <div class="creator-role">Project Contributor</div>
                <div class="creator-id">22BCS12846</div>
            </div>
            <div class="creator-card">
                <div class="creator-name">Abhishek Pratap Singh</div>
                <div class="creator-role">Project Contributor</div>
                <div class="creator-id">22BCS12922</div>
            </div>
            <div class="creator-card">
                <div class="creator-name">Atharva Deshmukh</div>
                <div class="creator-role">Project Contributor</div>
                <div class="creator-id">22BCS12930</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
