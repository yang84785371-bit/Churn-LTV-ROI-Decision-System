# src/decision/roi_backtest.py

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--churn_path", type=str,
                    default="outputs/models/churn/calibrated/scored_test.parquet")
    ap.add_argument("--ltv_path", type=str,
                    default="outputs/models/ltv/scored_test_ltv.parquet")
    ap.add_argument("--out_dir", type=str,
                    default="outputs/roi")
    ap.add_argument("--touch_cost", type=float, default=10.0)
    ap.add_argument("--uplift_rate", type=float, default=0.20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # -- 分别读取流失 以及ltv的数据 -- 
    churn = pd.read_parquet(args.churn_path)
    ltv = pd.read_parquet(args.ltv_path)
    # -- 将流失率以及召回预测收益 merge一下 --
    df = churn.merge(
        ltv[["user_id", "cutoff_week", "expected_margin_4w"]],
        on=["user_id", "cutoff_week"],
        how="inner"
    )

    # 使用校准后的流失概率
    df["p_churn"] = df["p_churn_cal"]

    # -- 策略分数：高流失 × 高价值 -- 
    df["target_score"] = df["p_churn"] * df["expected_margin_4w"] # 得分
    df["touch_cost"] = float(args.touch_cost) # 召回成本的话是自己设置的
    df["expected_uplift_gain"] = float(args.uplift_rate) * df["target_score"] # 我们假设干预可以挽回 20% 的风险暴露值 因为 plift_rate = 0.2
    df["expected_net_gain"] = df["expected_uplift_gain"] - df["touch_cost"] # 净收益=挽回收益−成本 如果是负数就不值得救了
    df = df.sort_values("target_score", ascending=False).reset_index(drop=True)
    # -- 准备一些容器 --
    results = []
    N = len(df)
    prev_net_gain = None
    prev_cost = None
    for frac in np.linspace(0.05, 0.50, 10): # 5 到 50生成 10个点 
        k = max(1, int(N * frac)) # 上面已经排序 就是去toppercent 的用户 
        sub = df.iloc[:k].copy() # 子集 基于这个进行

        total_cost = float(sub["touch_cost"].sum()) # 总的成本 
        net_gain = float(sub["expected_net_gain"].sum())    # 总的净收益
        roi = net_gain / (total_cost + 1e-9) # roi 就是 净收益与成本的比例
        net_gain_per_user = net_gain / max(1, len(sub)) # 平均净收益
        marginal_net_gain = None if prev_net_gain is None else (net_gain - prev_net_gain) # 同比净收益
        # -- 同比roi --
        marginal_roi = None if (prev_cost is None or prev_net_gain is None) else (net_gain - prev_net_gain) / ((total_cost - prev_cost) + 1e-9)
        prev_net_gain, prev_cost = net_gain, total_cost # 存起来 下一个周期用
        results.append({
            "target_frac": frac,
            "num_users": len(sub),
            "expected_net_gain": net_gain,
            "total_cost": total_cost,
            "roi": roi,
            "net_gain_per_user": net_gain_per_user,
            "marginal_net_gain": marginal_net_gain,
            "marginal_roi": marginal_roi,
        })

    roi_df = pd.DataFrame(results)
    roi_df.to_csv(out_dir / "roi_curve.csv", index=False)

    print("[OK] ROI curve saved to:", out_dir)
    print(roi_df.to_string(index=False))


if __name__ == "__main__":
    main()