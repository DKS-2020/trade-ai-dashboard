# Global Trade Intelligence Dashboard

This project turns a basic trade analysis notebook into a cleaner final-year project demo using `Streamlit`.

## What it includes

- Interactive filtering by year, reporter country, partner country, and trade flow
- KPI cards for fast summary insight
- Trend, ranking, heatmap, and distribution charts
- A prediction page for estimating trade value from year and country pair
- A presentation-friendly project framing section

## Dataset

The app expects a CSV with these required columns:

- `Year`
- `ReporterName`
- `PartnerName`
- `TradeValue in 1000 USD`

Optional:

- `TradeFlowName`

## Run locally

1. Place your dataset in the same folder as `app.py` and name it `trade_1988_2021.csv`
2. Or upload the CSV from the sidebar after the app starts
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

## Presentation angle

Use this as a dashboard-driven analytics project, not only as a machine learning notebook. A stronger pitch is:

`An interactive system for analyzing international trade patterns and estimating country-to-country trade value.`
