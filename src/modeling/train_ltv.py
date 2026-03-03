# src/modeling/train_ltv.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# -- load cfg 例如 horizon backlook这些超参数 --
def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -- 样本切分 切分为 trian valid test -- 
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

# -- 获取字段 --
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_cols = {"user_id", "cutoff_week", "churn_4w", "future_margin_4w"}
    return [c for c in df.columns if c not in drop_cols]

# -- 构造验证标准 --
def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }

# -- 选tok 的lift / 整体收益 --
def top_decile_lift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    overall = float(np.mean(y_true)) if len(y_true) else 0.0
    if overall <= 0:
        return float("nan")
    k = max(1, int(np.floor(len(y_pred) * 0.10)))
    idx = np.argsort(-y_pred)[:k]
    return float(np.mean(y_true[idx]) / overall)


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_path", type=str,
                    default="data/processed/feature_table/feature_table_online_retail.parquet")
    ap.add_argument("--windows_cfg", type=str, default="configs/windows.yml")
    ap.add_argument("--out_dir", type=str, default="outputs/models/ltv")
    ap.add_argument("--use_lightgbm", action="store_true", help="Use LightGBM models (recommended).")
    ap.add_argument("--positive_threshold", type=float, default=0.0,
                    help="Define 'positive future value' as future_margin_4w > threshold.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.windows_cfg))
    train_ratio = float(cfg.get("splits", {}).get("train_ratio", 0.70))
    valid_ratio = float(cfg.get("splits", {}).get("valid_ratio", 0.15))
    # -- mkdir一下output folder --
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # -- 读取数据
    df = pd.read_parquet(args.feature_path)
    df["cutoff_week"] = pd.to_datetime(df["cutoff_week"])
    # -- 提取特征 --
    feats = get_feature_cols(df)
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)

    # -- 指明标签 --
    y = df["future_margin_4w"].astype(float).values
    y_pos = (y > args.positive_threshold).astype(int)

    train_df, valid_df, test_df = time_split_by_cutoff_week(df, train_ratio, valid_ratio) # 数据集切分

    X_train, y_train, y_train_pos = train_df[feats], train_df["future_margin_4w"].astype(float).values, (train_df["future_margin_4w"].values > args.positive_threshold).astype(int) # 变量 切分为 feature 以及label


    X_valid, y_valid, y_valid_pos = valid_df[feats], valid_df["future_margin_4w"].astype(float).values, (valid_df["future_margin_4w"].values > args.positive_threshold).astype(int)
    X_test,  y_test,  y_test_pos  = test_df[feats],  test_df["future_margin_4w"].astype(float).values,  (test_df["future_margin_4w"].values > args.positive_threshold).astype(int)

    rows = [] # 进行记录

    '''
    Stage 1: P(value > 0)
    '''
    clf_name = "logistic"
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train_pos)
    p_valid = clf.predict_proba(X_valid)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    joblib.dump(clf, out_dir / f"stage1_{clf_name}.joblib")

    rows.append({
        "stage": "stage1",
        "model": clf_name,
        "split": "valid",
        "pos_rate": float(np.mean(y_valid_pos)),
        "avg_pred_pos": float(np.mean(p_valid)),
    })
    rows.append({
        "stage": "stage1",
        "model": clf_name,
        "split": "test",
        "pos_rate": float(np.mean(y_test_pos)),
        "avg_pred_pos": float(np.mean(p_test)),
    })

    if args.use_lightgbm:
        if not HAS_LGB:
            raise RuntimeError("lightgbm is not installed. Run: pip install lightgbm")
        clf_name = "lightgbm"
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            class_weight="balanced",
        )
        clf_lgb.fit(X_train, y_train_pos, eval_set=[(X_valid, y_valid_pos)], eval_metric="auc")
        p_valid_lgb = clf_lgb.predict_proba(X_valid)[:, 1]
        p_test_lgb = clf_lgb.predict_proba(X_test)[:, 1]
        joblib.dump(clf_lgb, out_dir / f"stage1_{clf_name}.joblib")

        rows.append({"stage": "stage1", "model": clf_name, "split": "valid",
                     "pos_rate": float(np.mean(y_valid_pos)), "avg_pred_pos": float(np.mean(p_valid_lgb))})
        rows.append({"stage": "stage1", "model": clf_name, "split": "test",
                     "pos_rate": float(np.mean(y_test_pos)), "avg_pred_pos": float(np.mean(p_test_lgb))})

    '''
    Stage 2: E[value | value > 0]
    '''
    train_pos = train_df[train_df["future_margin_4w"] > args.positive_threshold].copy()
    valid_pos = valid_df[valid_df["future_margin_4w"] > args.positive_threshold].copy()
    test_pos = test_df[test_df["future_margin_4w"] > args.positive_threshold].copy()

    X_train2, y_train2 = train_pos[feats], train_pos["future_margin_4w"].astype(float).values
    X_valid2, y_valid2 = valid_pos[feats], valid_pos["future_margin_4w"].astype(float).values
    X_test2,  y_test2  = test_pos[feats],  test_pos["future_margin_4w"].astype(float).values

    #  -- 进行估计 -- 
    if args.use_lightgbm:
        reg_name = "lightgbm"
        reg = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
        )
        reg.fit(X_train2, y_train2, eval_set=[(X_valid2, y_valid2)], eval_metric="l2")
    else:
        # -- 如果不用lightgbm就用简单方法 --
        reg_name = "mean_pos"
        reg = float(np.mean(y_train2)) if len(y_train2) else 0.0
    # -- 是否退化 --
    if isinstance(reg, float):
        pred_valid2 = np.full_like(y_valid2, reg, dtype=float)
        pred_test2 = np.full_like(y_test2, reg, dtype=float)
    else:
        pred_valid2 = reg.predict(X_valid2)
        pred_test2 = reg.predict(X_test2)

    # -- 估计值为0的话 进行截断 --
    pred_valid2 = np.clip(pred_valid2, 0, None)
    pred_test2 = np.clip(pred_test2, 0, None)

    if not isinstance(reg, float):
        joblib.dump(reg, out_dir / f"stage2_{reg_name}.joblib")

    m_valid2 = reg_metrics(y_valid2, pred_valid2) if len(y_valid2) else {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    m_test2 = reg_metrics(y_test2, pred_test2) if len(y_test2) else {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    rows.append({"stage": "stage2", "model": reg_name, "split": "valid",
                 **m_valid2, "top_decile_lift": top_decile_lift(y_valid2, pred_valid2)})
    rows.append({"stage": "stage2", "model": reg_name, "split": "test",
                 **m_test2, "top_decile_lift": top_decile_lift(y_test2, pred_test2)})

    '''
    Combine: E[value] = P(pos) * E[value|pos]
    一阶段 可以的话用lightgbm 不然就逻辑
    二阶段使用 lightgbm
    '''
    if args.use_lightgbm and HAS_LGB:
        stage1 = joblib.load(out_dir / "stage1_lightgbm.joblib")
        p_valid_pos = stage1.predict_proba(X_valid)[:, 1]
        p_test_pos = stage1.predict_proba(X_test)[:, 1]
    else:
        stage1 = joblib.load(out_dir / "stage1_logistic.joblib")
        p_valid_pos = stage1.predict_proba(X_valid)[:, 1]
        p_test_pos = stage1.predict_proba(X_test)[:, 1]

    # -- 我们对所有需要的样本进行预测 不筛选 --
    if isinstance(reg, float): # 第二阶段是否退化
        cond_valid = np.full(len(X_valid), reg, dtype=float) # 所有人都预测同一个条件均值
        cond_test = np.full(len(X_test), reg, dtype=float) # 
    else:
        cond_valid = np.clip(reg.predict(X_valid), 0, None) # 正常预测 将负数截断为 0
        cond_test = np.clip(reg.predict(X_test), 0, None)

    ev_valid = p_valid_pos * cond_valid
    ev_test = p_test_pos * cond_test

    # -- 计算指标 --
    comb_valid = reg_metrics(y_valid, ev_valid)
    comb_test = reg_metrics(y_test, ev_test)

    rows.append({"stage": "combined", "model": "E[value]=P*E|pos", "split": "valid",
                 **comb_valid, "top_decile_lift": top_decile_lift(y_valid, ev_valid)})
    rows.append({"stage": "combined", "model": "E[value]=P*E|pos", "split": "test",
                 **comb_test, "top_decile_lift": top_decile_lift(y_test, ev_test)})

    # -- 保存 score 为了 roi --
    scored_test = test_df[["user_id", "cutoff_week", "future_margin_4w", "churn_4w"]].copy()
    scored_test["p_pos_value"] = p_test_pos
    scored_test["cond_value_pred"] = cond_test
    scored_test["expected_margin_4w"] = ev_test
    scored_test.to_parquet(out_dir / "scored_test_ltv.parquet", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics_ltv.csv", index=False)

    print("[OK] saved models & metrics to:", out_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()