# src/eval/plot_roi_and_breakdown.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_path", type=str, default="outputs/roi/roi_curve_sim_strategies_by_scenario.csv")
    ap.add_argument("--scenario", type=str, default="base", help="base/harm/misalign/high_noise/saturation...")
    ap.add_argument("--strategy_prefix", type=str, default="", help="optional: filter strategies by prefix")
    ap.add_argument("--out_path", type=str, default="outputs/figures/roi_and_breakdown.png")
    args = ap.parse_args()

    df = pd.read_csv(args.roi_path)
    if "scenario" not in df.columns:
        raise ValueError("roi_path must be long-format with 'scenario' column.")

    df = df[df["scenario"] == args.scenario].copy()
    if df.empty:
        raise ValueError(f"No rows for scenario='{args.scenario}'")

    if args.strategy_prefix:
        df = df[df["strategy"].astype(str).str.startswith(args.strategy_prefix)].copy()
        if df.empty:
            raise ValueError(f"No rows after strategy_prefix='{args.strategy_prefix}'")

    # -- 确保大中小三券子啊段存在 -- 
    breakdown_cols = [
        "share_treated", "share_small", "share_medium", "share_large", "avg_cost_per_treated"
    ]
    have_breakdown = [c for c in breakdown_cols if c in df.columns]

    '''
        画图：
        上面是roi曲线 下面是发券结构 自变量都是frac
    '''
    nrows = 2 if have_breakdown else 1
    fig = plt.figure(figsize=(10, 6 if have_breakdown else 4))

    ax1 = fig.add_subplot(nrows, 1, 1)

    for strat, g in df.groupby("strategy", sort=True):
        g = g.sort_values("target_frac")
        ax1.plot(g["target_frac"], g["net_gain"], marker="o", label=strat)

    ax1.set_title(f"ROI Curve (net_gain) - scenario: {args.scenario}")
    ax1.set_xlabel("target_frac")
    ax1.set_ylabel("net_gain")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    if have_breakdown:
        ax2 = fig.add_subplot(nrows, 1, 2)
        # -- 只画 best_coupon 类策略的 breakdown 会更清晰--
        # -- 这里不强制，按过滤后的 df 直接画 --
        for strat, g in df.groupby("strategy", sort=True):
            g = g.sort_values("target_frac")
            if "avg_cost_per_treated" in have_breakdown:
                ax2.plot(g["target_frac"], g["avg_cost_per_treated"], marker=".", label=f"{strat} | avg_cost")
        # -- hares（可选） --
        if "share_large" in have_breakdown:
            for strat, g in df.groupby("strategy", sort=True):
                g = g.sort_values("target_frac")
                ax2.plot(g["target_frac"], g["share_large"], marker="x", label=f"{strat} | share_large")

        ax2.set_title("Policy breakdown (if available)")
        ax2.set_xlabel("target_frac")
        ax2.grid(True)
        ax2.legend(fontsize=8)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()