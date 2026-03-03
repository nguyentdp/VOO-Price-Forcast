# VOO Price Forecast (5-Day) — Ridge Regression

Portfolio project predicting the **5-trading-day ahead** price movement of **VOO (Vanguard S&P 500 ETF)** using engineered OHLCV features and a **Ridge Regression** model, compared against a **no-change baseline**.

## Live Demo (GitHub Pages)
After enabling Pages from `/docs`, your site will be:
`https://nguyentdp.github.io/VOO-Price-Forcast/`

## Highlights
- **Model:** Ridge Regression (L2 regularization)
- **Horizon:** 5 trading days ahead
- **Split:** time-based (no shuffle), last **252 trading days** as test
- **Feature engineering:** returns, moving average ratio, volatility, range/gap, log-volume
- **Plots:** actual vs predicted, residuals, top coefficients (Matplotlib)

## How to Run
### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Data
Place the CSV here:
```bash
data/VOO Stock Data.csv
```

## Train + Evaluate
```bash
python src/train.py
```

## Make Plots (Matplotlib)
```bash
python src/make_plots.py
```

## Outputs:
```bash
reports/figures/*.png
```

## Copy plots into the website assets
```bash
mkdir -p docs/assets
cp reports/figures/*.png docs/assets/
```

## Note on datasets

Many Kaggle datasets shouldn’t be redistributed publicly. Prefer keeping the CSV out of GitHub and providing download instructions.

## Quick local preview (before you rely on Pages)

```bash
cd docs
python -m http.server 8000
```

## Then open:
```bash
http://localhost:8000
```