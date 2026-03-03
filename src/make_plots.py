import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZON_DAYS = 5
TEST_DAYS = 252
CLIP_Z = 20.0


def winsorize(s: pd.Series, lower_q=0.005, upper_q=0.995) -> pd.Series:
    if s is None or s.dropna().empty:
        return s
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)


def parse_dates_safely(date_series: pd.Series) -> pd.Series:
    ds = date_series.astype(str).str.strip()
    ds10 = ds.str.slice(0, 10)

    pct_ymd = ds10.str.match(r"^\d{4}-\d{2}-\d{2}$").mean()
    pct_mdy = ds10.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").mean()

    if pct_ymd > 0.9:
        return pd.to_datetime(ds10, format="%Y-%m-%d", errors="coerce")
    if pct_mdy > 0.9:
        return pd.to_datetime(ds10, format="%m/%d/%Y", errors="coerce")

    try:
        return pd.to_datetime(ds, format="mixed", errors="coerce")
    except TypeError:
        return pd.to_datetime(ds, errors="coerce")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["price"] = df[price_col]

    df["ret_1"] = df["price"].pct_change(1)
    df["ret_2"] = df["price"].pct_change(2)
    df["ret_5"] = df["price"].pct_change(5)
    df["ret_10"] = df["price"].pct_change(10)

    df["ma_5"] = df["price"].rolling(5).mean()
    df["ma_10"] = df["price"].rolling(10).mean()
    df["ma_20"] = df["price"].rolling(20).mean()
    df["ma_ratio_5"] = df["price"] / df["ma_5"]

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["range"] = (df["High"] - df["Low"]) / df["price"].replace(0, np.nan)
    prev_price = df["price"].shift(1).replace(0, np.nan)
    df["gap"] = (df["Open"] - prev_price) / prev_price

    if "Volume" in df.columns:
        df["log_vol"] = np.log1p(df["Volume"])
        df["log_vol_diff"] = df["log_vol"].diff()
        df["log_vol_ma_20"] = df["log_vol"].rolling(20).mean()

    for col in ["ret_1", "gap", "range", "ma_ratio_5"]:
        df[col] = winsorize(df[col])
    if "log_vol_diff" in df.columns:
        df["log_vol_diff"] = winsorize(df["log_vol_diff"])

    df["y_ret_future"] = df["price"].pct_change(HORIZON_DAYS).shift(-HORIZON_DAYS)
    df["y_ret_future"] = winsorize(df["y_ret_future"])

    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    df = pd.read_csv("data/VOO Stock Data.csv")
    df.columns = df.columns.str.strip()

    df["Date"] = parse_dates_safely(df["Date"])
    df = df.dropna(subset=["Date"])

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = make_features(df)

    feature_cols = [
        "ret_1", "ret_2", "ret_5", "ret_10",
        "ma_ratio_5", "vol_10", "vol_20",
        "range", "gap",
        "log_vol", "log_vol_diff", "log_vol_ma_20"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price", "y_ret_future"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    X = df[feature_cols].astype(np.float64)
    y = df["y_ret_future"].astype(np.float64)

    n = len(df)
    n_test = min(TEST_DAYS, max(60, n // 5))

    X_train = X.iloc[:-n_test].to_numpy(dtype=np.float64)
    y_train = y.iloc[:-n_test].to_numpy(dtype=np.float64)

    X_test = X.iloc[-n_test:].to_numpy(dtype=np.float64)
    y_test = y.iloc[-n_test:].to_numpy(dtype=np.float64)

    test_price_today = df["price"].iloc[-n_test:].to_numpy(dtype=np.float64)
    test_future_price = df["price"].shift(-HORIZON_DAYS).iloc[-n_test:].to_numpy(dtype=np.float64)
    test_future_dates = df["Date"].shift(-HORIZON_DAYS).iloc[-n_test:]

    clipper = FunctionTransformer(lambda A: np.clip(A, -CLIP_Z, CLIP_Z))

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(threshold=1e-6)),
        ("scale", StandardScaler()),
        ("clip", clipper),
        ("linreg", LinearRegression()),
    ])

    model.fit(X_train, y_train)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        y_pred_ret = model.predict(X_test)
    y_pred_ret = np.clip(y_pred_ret, -0.20, 0.20)

    pred_future_price = test_price_today * (1.0 + y_pred_ret)

    valid = np.isfinite(test_future_price) & test_future_dates.notna().to_numpy()
    dates = test_future_dates[valid]
    y_true_price = test_future_price[valid]
    y_pred_price = pred_future_price[valid]
    y_true_ret = y_test[valid]
    y_pred_ret_valid = y_pred_ret[valid]

    mae = mean_absolute_error(y_true_price, y_pred_price)
    rmse = mean_squared_error(y_true_price, y_pred_price) ** 0.5
    dir_acc = ((y_pred_ret_valid > 0) == (y_true_ret > 0)).mean()

    baseline_price = test_price_today[valid]
    base_mae = mean_absolute_error(y_true_price, baseline_price)
    base_rmse = mean_squared_error(y_true_price, baseline_price) ** 0.5
    base_dir = ((False) == (y_true_ret > 0)).mean()

    print("Linear Regression vs Baseline (for plots):")
    print(f"  MAE  {mae:.4f} vs {base_mae:.4f}")
    print(f"  RMSE {rmse:.4f} vs {base_rmse:.4f}")
    print(f"  Dir  {dir_acc:.3f} vs {base_dir:.3f}")

    outdir = "reports/figures"
    ensure_dir(outdir)

    plt.figure()
    plt.plot(dates, y_true_price, label="Actual")
    plt.plot(dates, y_pred_price, label="Predicted")
    plt.title(f"VOO: Actual vs Predicted Price ({HORIZON_DAYS}-day ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "actual_vs_pred.png"), dpi=200)
    plt.close()

    residuals = y_true_price - y_pred_price
    plt.figure()
    plt.plot(dates, residuals)
    plt.title("Residuals Over Time (Actual - Predicted)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "residuals.png"), dpi=200)
    plt.close()

    linreg = model.named_steps["linreg"]
    vt = model.named_steps["vt"]
    support = vt.get_support()
    kept_features = [f for f, keep in zip(feature_cols, support) if keep]
    coefs = linreg.coef_

    k = min(10, len(coefs))
    top_idx = np.argsort(np.abs(coefs))[::-1][:k]
    top_feats = [kept_features[i] for i in top_idx]
    top_coefs = coefs[top_idx]

    plt.figure()
    plt.bar(top_feats, top_coefs)
    plt.title("Top Linear Regression Coefficients (Signed)")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "top_coefficients.png"), dpi=200)
    plt.close()

    print(f"Saved plots to {outdir}/:")
    print(" - actual_vs_pred.png")
    print(" - residuals.png")
    print(" - top_coefficients.png")


if __name__ == "__main__":
    main()