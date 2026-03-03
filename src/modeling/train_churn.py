# src/modeling/train_churn.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# -- 加载配置文件 --
def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -- 切分样本 --
def time_split_by_cutoff_week(df: pd.DataFrame, train_ratio: float, valid_ratio: float):
    """
    Split by unique cutoff_week (time-based).
    """
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

# -- 特征名 --
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    # exclude identifiers + labels
    drop_cols = {"user_id", "cutoff_week", "churn_4w", "future_margin_4w"}
    feats = [c for c in df.columns if c not in drop_cols]
    return feats

# -- auc defensiv -- 
def eval_binary(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {}
    # -- 如果只有一个类别 那auc会失效 -- 
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:# -- 如果失效我们就令他变成 nan -- 
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out

# -- 计算召回率 --
def topk_recall(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10) -> float:
    n = len(y_true)
    k = max(1, int(np.floor(n * top_frac)))
    idx = np.argsort(-y_prob)[:k]
    pos = (y_true == 1)
    denom = pos.sum()
    if denom == 0:
        return float("nan")
    return float((y_true[idx] == 1).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_path", type=str,
                    default="data/processed/feature_table/feature_table_online_retail.parquet")
    ap.add_argument("--windows_cfg", type=str, default="configs/windows.yml")
    ap.add_argument("--out_dir", type=str, default="outputs/models/churn")
    ap.add_argument("--use_lightgbm", action="store_true",
                    help="Train LightGBM model (recommended). Requires lightgbm installed.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.windows_cfg))
    train_ratio = float(cfg.get("splits", {}).get("train_ratio", 0.70))
    valid_ratio = float(cfg.get("splits", {}).get("valid_ratio", 0.15))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.feature_path)
    df["cutoff_week"] = pd.to_datetime(df["cutoff_week"])

    feature_cols = get_feature_cols(df)
    # -- defensive --
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    # -- 分出训练集与数据集 --
    train_df, valid_df, test_df = time_split_by_cutoff_week(df, train_ratio, valid_ratio)
    # -- 分特征与标签 --
    X_train, y_train = train_df[feature_cols], train_df["churn_4w"].astype(int).values
    X_valid, y_valid = valid_df[feature_cols], valid_df["churn_4w"].astype(int).values
    X_test, y_test = test_df[feature_cols], test_df["churn_4w"].astype(int).values

    # --- 逻辑回归作为baseline ---
    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1))
    ]) # 初始化一个逻辑模型
    logit.fit(X_train, y_train) # 拟合
    p_valid_logit = logit.predict_proba(X_valid)[:, 1] # 进行预测
    p_test_logit = logit.predict_proba(X_test)[:, 1] # 进行预测 
    # -- 计算指标 --
    metrics = []
    m_valid = eval_binary(y_valid, p_valid_logit) # 计算auc
    m_test = eval_binary(y_test, p_test_logit)
    metrics.append({
        "model": "logistic",
        "split": "valid",
        **m_valid,
        "top10_recall": topk_recall(y_valid, p_valid_logit, 0.10)
    })
    metrics.append({
        "model": "logistic",
        "split": "test",
        **m_test,
        "top10_recall": topk_recall(y_test, p_test_logit, 0.10)
    })

    joblib.dump(logit, out_dir / "logistic.joblib")

    # -- lightgbm作为候选 --
    if args.use_lightgbm:
        if not HAS_LGB:
            raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm")
        # -- 初始化一个模型 --
        lgbm = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            class_weight="balanced",
        )
        lgbm.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc"
        )
        p_valid_lgb = lgbm.predict_proba(X_valid)[:, 1]
        p_test_lgb = lgbm.predict_proba(X_test)[:, 1]

        m_valid = eval_binary(y_valid, p_valid_lgb)
        m_test = eval_binary(y_test, p_test_lgb)
        metrics.append({
            "model": "lightgbm",
            "split": "valid",
            **m_valid,
            "top10_recall": topk_recall(y_valid, p_valid_lgb, 0.10)
        })
        metrics.append({
            "model": "lightgbm",
            "split": "test",
            **m_test,
            "top10_recall": topk_recall(y_test, p_test_lgb, 0.10)
        })

        joblib.dump(lgbm, out_dir / "lightgbm.joblib")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics_churn.csv", index=False)

    
    print("[OK] saved models & metrics to:", out_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()