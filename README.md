# VOO Price Forecast (5-Day) — Linear Regression

Portfolio project that predicts the **5-trading-day ahead** price movement of **VOO (Vanguard S&P 500 ETF)** using engineered OHLCV features and **Linear Regression**, compared against a simple **no-change baseline**.

> Educational project only — not financial advice.

## Live Portfolio Page (GitHub Pages)
If Pages is enabled from `/docs`, your site URL will look like:
- `https://nguyentdp.github.io/VOO-Price-Forcast/`

(You can also copy the exact URL shown in **Settings → Pages**.)

---

## What this project shows
- A complete supervised learning workflow (end-to-end)
- Feature engineering (not using raw data “as-is”)
- A fair time-based train/test split (no shuffling)
- Baseline comparison (so results are meaningful)
- Clear Matplotlib plots for a portfolio page

---

## Data (OHLCV)
The dataset is historical daily trading data:

- **Open**: price at the start of the day  
- **High**: highest price that day  
- **Low**: lowest price that day  
- **Close**: price at the end of the day  
- **Volume**: how much was traded that day  

---

## Reproduce

### 1) Environment
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
python -m http.server 8000 --directory docs
```

## Then open:
```bash
http://localhost:8000
```