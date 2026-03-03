# src/modeling/calibrate.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

# -- 切分 因为我们这里还要进行一次训练 所以我们再切分一次 毕竟train churn训练完成切分就是无用的 --
def time_split_by_cutoff_week(df: pd.DataFrame, train_ratio=0.70, valid_ratio=0.15):
    weeks = np.array(sorted(df["cutoff_week"].unique()))
    n = len(weeks)
    n_train = int(np.floor(n * train_ratio))
    n_valid = int(np.floor(n * valid_ratio))
    train_weeks = set(weeks[:n_train])
    valid_weeks = set(weeks[n_train:n_train + n_valid])
    test_weeks = set(weeks[n_train + n_valid:])

    train = df[df["cutoff_week"].isin(train_weeks)].copy()
    valid = df[df["cutoff_week"].isin(valid_weeks)].copy()
    test = df[df["cutoff_week"].isin(test_weeks)].copy()
    return train, valid, test

# -- 获得字段名 --
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = {"user_id", "cutoff_week", "churn_4w", "future_margin_4w"}
    return [c for c in df.columns if c not in drop_cols]

# -- 计算指标 --
def metrics(y, p):
    out = {}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, p))
    return out

# -- 生成概率校准曲线的数据表 --
def calibration_curve_df(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    # -- 按照预测概率分桶 --
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_prob, q)
    edges[0] = 0.0
    edges[-1] = 1.0

    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue
        rows.append({
            "bin": i,
            "p_pred_mean": float(np.mean(y_prob[mask])),
            "p_true_mean": float(np.mean(y_true[mask])),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_path", type=str,
                    default="data/processed/feature_table/feature_table_online_retail.parquet")
    ap.add_argument("--model_path", type=str,
                    default="outputs/models/churn/lightgbm.joblib")
    ap.add_argument("--out_dir", type=str,
                    default="outputs/models/churn/calibrated")
    ap.add_argument("--method", type=str, default="isotonic", choices=["isotonic", "sigmoid"])
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # -- 读取特征 --
    df = pd.read_parquet(args.feature_path)
    df["cutoff_week"] = pd.to_datetime(df["cutoff_week"])# 确认类型
    # --字段名 用于防御 --
    feats = get_feature_cols(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0) # 防御
    # -- 这里用不到train 是符合概率校正的逻辑的 --
    train_df, valid_df, test_df = time_split_by_cutoff_week(df) # 再进行一次切分 
    # -- 区分特征与标签 -- 
    X_valid, y_valid = valid_df[feats], valid_df["churn_4w"].astype(int).values
    X_test, y_test = test_df[feats], test_df["churn_4w"].astype(int).values
    # -- load churn model -- 
    base_model = joblib.load(args.model_path)

    # -- 使用 train的模型得到valid 以及 test 概率 -- 
    p_valid_base = base_model.predict_proba(X_valid)[:, 1]
    p_test_base = base_model.predict_proba(X_test)[:, 1]
    #  -- 概率校正 --
    if args.method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip") # 确定cal模型 
        calibrator.fit(p_valid_base, y_valid) # 训练更灵活的模型
        p_valid_cal = calibrator.predict(p_valid_base) # 进行校正
        p_test_cal = calibrator.predict(p_test_base) 
    # -- 这里用sigmoid模型 --
    elif args.method == "sigmoid":
        lr = LogisticRegression()
        lr.fit(p_valid_base.reshape(-1, 1), y_valid)
        p_valid_cal = lr.predict_proba(p_valid_base.reshape(-1, 1))[:, 1]
        p_test_cal = lr.predict_proba(p_test_base.reshape(-1, 1))[:, 1]
        calibrator = lr

    # -- 指标 --
    base_valid_m = metrics(y_valid, p_valid_base)
    base_test_m = metrics(y_test, p_test_base)
    cal_valid_m = metrics(y_valid, p_valid_cal)
    cal_test_m = metrics(y_test, p_test_cal)

    metrics_df = pd.DataFrame([
        {"model": "base", "split": "valid", **base_valid_m},
        {"model": "base", "split": "test", **base_test_m},
        {"model": f"cal_{args.method}", "split": "valid", **cal_valid_m},
        {"model": f"cal_{args.method}", "split": "test", **cal_test_m},
    ])
    metrics_df.to_csv(out_dir / "metrics_calibration.csv", index=False)

    # -- 校准曲线报告 --
    curve_valid_base = calibration_curve_df(y_valid, p_valid_base, n_bins=args.n_bins)
    curve_valid_cal = calibration_curve_df(y_valid, p_valid_cal, n_bins=args.n_bins)
    curve_test_base = calibration_curve_df(y_test, p_test_base, n_bins=args.n_bins)
    curve_test_cal = calibration_curve_df(y_test, p_test_cal, n_bins=args.n_bins)

    curve_valid_base.to_csv(out_dir / "cal_curve_valid_base.csv", index=False)
    curve_valid_cal.to_csv(out_dir / "cal_curve_valid_cal.csv", index=False)
    curve_test_base.to_csv(out_dir / "cal_curve_test_base.csv", index=False)
    curve_test_cal.to_csv(out_dir / "cal_curve_test_cal.csv", index=False)

    # -- save --
    joblib.dump(calibrator, out_dir / f"calibrator_{args.method}.joblib")

    '''
        把模型预测结果和真实标签整理好
        存成一个标准接口文件
        供后续决策模块使用
    '''
    scored_test = test_df[["user_id", "cutoff_week", "churn_4w", "future_margin_4w"]].copy() # 保留test的标签字段
    scored_test["p_churn_base"] = p_test_base
    scored_test["p_churn_cal"] = p_test_cal
    scored_test.to_parquet(out_dir / "scored_test.parquet", index=False)

    print("[OK] saved to:", out_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()