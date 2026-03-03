# src/features/build_features.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -- load 配置文件 -- 
def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -- 确保是按照userid week 进行排序 --
def ensure_weekly_sorted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["user_id", "week_start"]).reset_index(drop=True)
    return df

# -- 构造特征 --
'''
    主要分为7类：
        rfm
        交易质量
        趋势与波动
        节奏类
        触达与营销（仅接口）
        生命周期类
'''
def build_feature_table(
    events_weekly: pd.DataFrame,
    lookback_weeks: int,
    horizon_weeks: int,
) -> pd.DataFrame:
    '''
        滚动样本生成规则：
            每行 = user_id * cutoof_week 
            特征 ：根据[cutoff - lookback_weeks, cutoff)的数据进行聚合
            标签 :根据[cutoff, cutoff + horizon_weeks)进行聚合
    '''
    # -- 排序 --
    df = ensure_weekly_sorted(events_weekly)

    # -- 创造个性化的时间网格 -- 
    span = (
        df.groupby("user_id", as_index=False)["week_start"]
        .agg(first_week="min", last_week="max")
    )

    grids = []
    for r in span.itertuples(index=False):
        uid = r.user_id
        first_w = pd.to_datetime(r.first_week)
        last_w = pd.to_datetime(r.last_week)

        # extend to allow horizon labels; otherwise last cutoffs cannot see future window
        end_w = last_w + pd.Timedelta(weeks=horizon_weeks)
        weeks = pd.date_range(first_w, end_w, freq="W-MON")

        grids.append(pd.DataFrame({"user_id": uid, "week_start": weeks}))

    grid = pd.concat(grids, ignore_index=True) # 得到所有的个性化时间网格 

    # -- 连接回去df --
    df = grid.merge(df, on=["user_id", "week_start"], how="left")

    # -- 对于没交易的需要量化清洗一下 --
    fill_zero_cols = ["active", "orders", "items", "revenue", "refund", "margin", "promo_flag", "touch_flag", "touch_cost","net_revenue_weekly","refund_amount_weekly","sales_amount_weekly"]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            df[c] = 0
    df = df.merge(span[["user_id", "first_week", "last_week"]], on="user_id", how="left")
    df["active"] = df["active"].astype(int)
    df = df.sort_values(["user_id", "week_start"]).reset_index(drop=True)

    # -- 计算用户生命周期 -- 
    df["t"] = df.groupby("user_id").cumcount()

    min_t = lookback_weeks

    # -- 掐头去尾流出预测的用于预测以及真实结果的部分 --
    max_cutoff_week = df["last_week"] - pd.Timedelta(weeks=horizon_weeks)

    df["is_valid_cutoff"] = (df["t"] >= min_t) & (df["week_start"] <= max_cutoff_week)
    # -- 时间表 --
    cutoffs = df.loc[df["is_valid_cutoff"], ["user_id", "week_start", "t"]].rename(
        columns={"week_start": "cutoff_week", "t": "cutoff_t"}
    )
    # -- 第一次出现时间 --
    first_week_map = df.groupby("user_id")["week_start"].min().to_dict()
    out_rows = []
    # -- 分小表 避免跨 user --
    for uid, g in df.groupby("user_id", sort=False):
        g = g.reset_index(drop=True)
        L = len(g)
        first_week = first_week_map[uid]
        # -- 取出该id的有效时间 --
        c = cutoffs[cutoffs["user_id"] == uid]
        if c.empty:
            continue
        # -- 确认类型 --
        active = g["active"].to_numpy()
        orders = g["orders"].to_numpy(dtype=float)
        revenue = g["revenue"].to_numpy(dtype=float)
        margin = g["margin"].to_numpy(dtype=float)
        refund = g["refund"].to_numpy(dtype=float)
        items = g["items"].to_numpy(dtype=float)
        promo = g["promo_flag"].to_numpy(dtype=float)
        touch = g["touch_flag"].to_numpy(dtype=float)
        touch_cost = g["touch_cost"].to_numpy(dtype=float)
        # -- 预先求好总数 最后减就可以了 -- 
        def ps(x: np.ndarray) -> np.ndarray:
            return np.concatenate([[0.0], np.cumsum(x)])

        ps_active = ps(active)
        ps_orders = ps(orders)
        ps_revenue = ps(revenue)
        ps_margin = ps(margin)
        ps_refund = ps(refund)
        ps_items = ps(items)
        ps_promo = ps(promo)
        ps_touch = ps(touch)
        ps_touch_cost = ps(touch_cost)
        # -- 求sum in [l, r) 通过ps --
        def wsum(prefix: np.ndarray, l: int, r: int) -> float:
            return float(prefix[r] - prefix[l])
        # -- 这里开始求特征 --
        for _, row in c.iterrows():
            t0 = int(row["cutoff_t"]) # cutoff week 的周数
            cutoff_week = row["cutoff_week"]#cutoff week

            # -- feature窗口 -- 
            l1, r1 = t0 - lookback_weeks, t0
            # -- label 窗口 --
            l2, r2 = t0, t0 + horizon_weeks

            # -- labels --
            future_active_sum = wsum(ps_active, l2, r2)
            churn = 1 if future_active_sum == 0 else 0
            future_margin_sum = wsum(ps_margin, l2, r2)

            # -- features (basic RFM + quality) --
            lb_active_weeks = wsum(ps_active, l1, r1) # 过去 8 周里活跃了多少周？
            lb_orders = wsum(ps_orders, l1, r1) # 过去 8 周一共下了多少单？
            lb_revenue = wsum(ps_revenue, l1, r1)#过去 8 周总收入是多少
            lb_margin = wsum(ps_margin, l1, r1)#过去 8 周总利润是多少
            lb_refund = wsum(ps_refund, l1, r1)#过去 8 周总退款是多少
            lb_items = wsum(ps_items, l1, r1)# 过去 8 周买了多少件商品？
            lb_promo_weeks = wsum(ps_promo, l1, r1) # 过去 8 周有多少周参与过促销？
            lb_touch_weeks = wsum(ps_touch, l1, r1) # 过去 8 周有多少周被触达？
            lb_touch_cost = wsum(ps_touch_cost, l1, r1) # 过去 8 周一共花了多少触达成本？

            items_per_order = (lb_items / lb_orders) if lb_orders > 0 else 0.0 # 每单平均买多少件？
            rev_per_active_week = (lb_revenue / lb_active_weeks) if lb_active_weeks > 0 else 0.0 # 每个活跃周平均收入多少？
            touch_cost_per_active = (lb_touch_cost / lb_active_weeks) if lb_active_weeks > 0 else 0.0 # 每个活跃周平均触达成本？
            touch_cost_per_order = (lb_touch_cost / lb_orders) if lb_orders > 0 else 0.0 # 每单平均触达成本？
            last_active_idx = np.where(active[l1:r1] > 0)[0]
            if len(last_active_idx) == 0:
                recency_weeks = lookback_weeks # 距离最近一次活跃已经多少周？
            else:
                last_pos = last_active_idx[-1]  # 0..lookback-1
                recency_weeks = (lookback_weeks - 1 - int(last_pos))

            # -- 每单平均消费金额？ --
            aov = (lb_revenue / lb_orders) if lb_orders > 0 else 0.0
            refund_rate = (lb_refund / (lb_refund + max(lb_revenue, 0.0))) if (lb_refund + max(lb_revenue, 0.0)) > 0 else 0.0#退款占总金额的比例？
            # -- 这个用户从第一次出现到现在活了多少周？ --
            tenure_weeks = (pd.to_datetime(cutoff_week) - pd.to_datetime(first_week)).days // 7

            # -- 窗口利润 -- 
            margin_window = margin[l1:r1]
            # -- 窗口活跃 -- 
            active_window = active[l1:r1]

            # -- 连续多少周没活跃？-- 
            inactive_streak = 0
            for val in reversed(active_window):
                if val == 0:
                    inactive_streak += 1
                else:
                    break

            # -- 过去 8 周里最长连续活跃多少周？-- 
            longest_burst = 0
            current_burst = 0
            for val in active_window:
                if val > 0:
                    current_burst += 1
                    longest_burst = max(longest_burst, current_burst)
                else:
                    current_burst = 0

            # -- 利润是在上升还是下降？-- 
            rev_window = revenue[l1:r1]
            if len(rev_window) > 1 and np.mean(rev_window) > 0:
                revenue_cv = float(np.std(rev_window) / (np.mean(rev_window) + 1e-6))
            else:
                revenue_cv = 0.0
            if len(margin_window) > 1:
                x = np.arange(len(margin_window))
                slope = np.polyfit(x, margin_window, 1)[0]
            else:
                slope = 0.0

            w_short = min(4, lookback_weeks)
            l0, r0 = t0 - w_short, t0
            # -- 短期动量类 --
            rev_4w = wsum(ps_revenue, l0, r0) # 最近 4 周收入是多少？
            ord_4w = wsum(ps_orders, l0, r0) # 最近 4 周订单数是多少？
            act_4w = wsum(ps_active, l0, r0) # 最近 4 周活跃周数是多少？

            rev_mom = rev_4w / (lb_revenue / (lookback_weeks / w_short) + 1e-6) # 最近 4 周收入 vs 过去 8 周平均收入
            active_share_4w = act_4w / (lb_active_weeks + 1e-6) # 最近 4 周活跃占过去 8 周活跃的比例

            out_rows.append(
                {
                    "user_id": uid,
                    "cutoff_week": cutoff_week,
                    # labels
                    "churn_4w": churn,
                    "future_margin_4w": future_margin_sum,
                    # features
                    "recency_weeks_8w": recency_weeks,
                    "active_weeks_8w": lb_active_weeks,
                    "orders_8w": lb_orders,
                    "revenue_8w": lb_revenue,
                    "margin_8w": lb_margin,
                    "refund_8w": lb_refund,
                    "aov_8w": aov,
                    "refund_rate_8w": refund_rate,
                    "tenure_weeks": tenure_weeks,
                    "margin_slope_8w": slope,
                    "items_8w": lb_items,
                    "items_per_order_8w": items_per_order,
                    "rev_per_active_week_8w": rev_per_active_week,

                    "promo_weeks_8w": lb_promo_weeks,
                    "touch_weeks_8w": lb_touch_weeks,
                    "touch_cost_8w": lb_touch_cost,
                    "touch_cost_per_active_8w": touch_cost_per_active,
                    "touch_cost_per_order_8w": touch_cost_per_order,

                    "revenue_4w": rev_4w,
                    "orders_4w": ord_4w,
                    "active_weeks_4w": act_4w,
                    "revenue_mom_4w_vs_8w": rev_mom,
                    "active_share_4w": active_share_4w,

                    "inactive_streak_8w": inactive_streak,
                    "active_burst_8w": longest_burst,
                    "revenue_cv_8w": revenue_cv,
                }
            )

    feat = pd.DataFrame(out_rows)
    feat["cutoff_week"] = pd.to_datetime(feat["cutoff_week"])
    feat = feat.sort_values(["cutoff_week", "user_id"]).reset_index(drop=True)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_path", type=str,
                    default="data/processed/events_weekly/events_weekly_online_retail.parquet")
    ap.add_argument("--out_path", type=str,
                    default="data/processed/feature_table/feature_table_online_retail.parquet")
    ap.add_argument("--windows_cfg", type=str, default="configs/windows.yml")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.windows_cfg))
    lookback = int(cfg.get("lookback_weeks", 8))
    horizon = int(cfg.get("horizon_weeks", 4))

    events_path = Path(args.events_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(events_path)
    feat = build_feature_table(df, lookback_weeks=lookback, horizon_weeks=horizon)

    feat.to_parquet(out_path, index=False)
    preview_path = out_path.with_suffix(".csv")
    feat.head(2000).to_csv(preview_path, index=False)

    print("[OK] saved:", out_path)
    print("[OK] preview:", preview_path)
    print("[INFO] rows:", len(feat), "users:", feat["user_id"].nunique(),
          "cutoffs:", feat["cutoff_week"].nunique())
    print("[INFO] churn_rate:", round(feat["churn_4w"].mean(), 4),
          "future_margin_mean:", round(feat["future_margin_4w"].mean(), 4))


if __name__ == "__main__":
    main()