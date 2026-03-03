# src/adapters/make_events_online_retail.py
from __future__ import annotations
import json
import argparse
from pathlib import Path

import pandas as pd
import yaml

# -- load 配置文件 --
def _load_windows_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -- 读取raw数据 -- 
def _read_any_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path, engine="openpyxl")
    if suf in [".csv"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {suf} (use .xlsx/.xls/.csv)")

# -- 转换成周级数据 --
def _to_monday_week_start(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce")
    # -- 时间归为周一 --
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def build_events_weekly(
    raw_path: Path,
    out_path: Path,
    margin_rate: float,
    keep_negative_revenue: bool = False,
) -> pd.DataFrame:
    df = _read_any_table(raw_path)
    rename_map = {
        "Invoice": "InvoiceNo",
        "InvoiceNo": "InvoiceNo",
        "UnitPrice": "UnitPrice",
        "Price": "UnitPrice",
        "CustomerID": "CustomerID",
        "Customer ID": "CustomerID",
        "InvoiceDate": "InvoiceDate",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # -- 字段防御 --
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got columns: {list(df.columns)[:20]} ...")

    # -- 清洗数据 --
    '''
        必要字段drop na
        进行类型确定
        类型无法确定的dropna一下
    '''
    df = df.dropna(subset=["CustomerID", "InvoiceDate"]) 
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"])

    # -- 过滤掉无理数据 --
    df = df[df["UnitPrice"] > 0]
    df = df[df["Quantity"] != 0]

    # -- 转换为周时间 --
    df["week_start"] = _to_monday_week_start(df["InvoiceDate"])
    # -- 防御性确认 防止后续口径失误 --
    # -- pandas weekday: Monday=0 ... Sunday=6 --
    bad = df[df["week_start"].dt.weekday != 0]
    if not bad.empty:
        print("[WARN] Found week_start not on Monday. Showing examples:")
        print(bad[["InvoiceDate", "week_start"]].head(5).to_string(index=False))

    # -- 数量*客单价得到收益 --
    df["line_amount"] = df["Quantity"] * df["UnitPrice"]

    # -- 是否退款 --
    inv = df["InvoiceNo"].astype(str)
    df["is_refund"] = (df["Quantity"] < 0) | (inv.str.startswith("C"))

    df["refund_amount"] = df["line_amount"].where(df["is_refund"], 0.0).abs() # 退款损失金额
    df["sales_amount"] = df["line_amount"].where(~df["is_refund"], 0.0) # 销售额

    # -- 收益 = 销售额 - 退款额 -- 
    df["net_revenue"] = df["sales_amount"] - df["refund_amount"]

    # -- 按照用户id 以及周 进行聚合 --
    g = df.groupby(["CustomerID", "week_start"], as_index=False)
    weekly = g.agg(
        orders=("InvoiceNo", "nunique"),
        items=("Quantity", "sum"),
        net_revenue_weekly=("net_revenue", "sum"),
        refund_amount_weekly=("refund_amount", "sum"),
        sales_amount_weekly=("sales_amount", "sum"),
    )
    # -- 是否进行clip --
    if keep_negative_revenue:
        weekly["revenue"] = weekly["net_revenue_weekly"]
    else:
        weekly["revenue"] = weekly["net_revenue_weekly"].clip(lower=0.0)

    weekly.rename(columns={"CustomerID": "user_id"}, inplace=True)

    # -- 是否活跃的bool值 --
    weekly["active"] = (weekly["orders"] > 0).astype(int)

    # -- 默认的promo 和touch 数据 --
    weekly["promo_flag"] = 0
    weekly["touch_flag"] = 0
    weekly["touch_cost"] = 0.0

    # -- 利润近似 --
    weekly["margin"] = (weekly["revenue"] * float(margin_rate)).astype(float)

    # -- 排序 --
    weekly = weekly.sort_values(["user_id", "week_start"]).reset_index(drop=True)
    return weekly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", type=str, default="data/raw/online_retail.xlsx",
                    help="Path to Online Retail raw file (.xlsx/.csv).")
    ap.add_argument("--out_path", type=str, default="data/processed/events_weekly/events_weekly_online_retail.parquet")
    ap.add_argument("--windows_cfg", type=str, default="configs/windows.yml")
    ap.add_argument("--keep_negative_revenue", action="store_true",
                    help="Keep weekly net revenue possibly negative. Default: keep net revenue as-is.")
    args = ap.parse_args()

    cfg = _load_windows_cfg(Path(args.windows_cfg))
    margin_rate = float(cfg.get("margin", {}).get("margin_rate_default", 0.30))

    raw_path = Path(args.raw_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weekly = build_events_weekly(
        raw_path=raw_path,
        out_path=out_path,
        margin_rate=margin_rate,
        keep_negative_revenue=args.keep_negative_revenue,
    )

    # -- 数据质量报告 --
    total_sales = float(weekly["sales_amount_weekly"].sum())
    total_refund = float(weekly["refund_amount_weekly"].sum())
    refund_share = (total_refund / total_sales) if total_sales > 0 else float("nan")

    neg_net_share = float((weekly["net_revenue_weekly"] < 0).mean()) if "net_revenue_weekly" in weekly.columns else float("nan")
    zero_revenue_share = float((weekly["revenue"] <= 0).mean()) if "revenue" in weekly.columns else float("nan")

    print("\n[DATA QUALITY] events_weekly summary")
    print(f"rows(user-week): {len(weekly):,}")
    if {"user_id", "week_start"}.issubset(set(weekly.columns)):
        print(f"unique users: {weekly['user_id'].nunique():,} | unique weeks: {weekly['week_start'].nunique():,}")
    print(f"total_sales: {total_sales:,.2f} | total_refund: {total_refund:,.2f} | refund_share: {refund_share:.4f}")
    print(f"net_revenue_weekly < 0 share: {neg_net_share:.4f}")
    print(f"revenue <= 0 share: {zero_revenue_share:.4f}\n")

    # -- 保存数据质量报告 -- 
    quality_report = {
        "n_user_weeks": int(len(weekly)),
        "n_users": int(weekly["user_id"].nunique()) if "user_id" in weekly.columns else None,
        "n_weeks": int(weekly["week_start"].nunique()) if "week_start" in weekly.columns else None,
        "total_sales": total_sales,
        "total_refund": total_refund,
        "refund_share": refund_share,
        "neg_net_revenue_share": neg_net_share,
        "zero_revenue_share": zero_revenue_share,
    }

    quality_path = out_path.parent / "events_weekly_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality_report, f, indent=2)

    print("[OK] saved quality report to:", quality_path)

    # -- 整体字段的snapshot --
    schema_df = (
        weekly.dtypes.astype(str)
        .reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
    )
    out_schema = out_path.parent / "events_weekly_schema.csv"
    schema_df.to_csv(out_schema, index=False)
    print("[OK] saved schema to:", out_schema)


    weekly.to_parquet(out_path, index=False)
    # -- 保存 小数据csv 大数据parquet --
    preview_path = out_path.with_suffix(".csv")
    weekly.head(2000).to_csv(preview_path, index=False)

    print("[OK] saved:", out_path)
    print("[OK] preview:", preview_path)
    print("[INFO] rows:", len(weekly), "users:", weekly["user_id"].nunique(),
          "weeks:", weekly["week_start"].nunique())


if __name__ == "__main__":
    main()