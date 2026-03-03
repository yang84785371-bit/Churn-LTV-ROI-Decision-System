# src/decision/plot_roi_curve.py
# 支持 multi-scenario + long-format ROI 表

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roi_path",
        type=str,
        default="outputs/roi/roi_curve_sim_strategies_by_scenario.csv",
        help="Path to ROI curve csv (long format with scenario column).",
    )
    ap.add_argument(
        "--scenario",
        type=str,
        default="base",
        help="Which scenario to plot (e.g. base/harm/...).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="net_gain",
        choices=["net_gain", "roi", "inc_gain"],
        help="Metric to plot on y-axis.",
    )
    ap.add_argument(
        "--strategy_prefix",
        type=str,
        default="",
        help="Optional: only plot strategies whose name starts with this prefix.",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default="outputs/roi/roi_curve_plot.png",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.roi_path)

    if "scenario" not in df.columns:
        raise ValueError("ROI file must contain 'scenario' column (multi-scenario long format).")

    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not in columns: {df.columns.tolist()}")

    # -- 过滤场景 -- 
    df = df[df["scenario"] == args.scenario].copy()
    if df.empty:
        raise ValueError(
            f"No rows for scenario='{args.scenario}'. "
            f"Available scenarios: {sorted(pd.read_csv(args.roi_path)['scenario'].unique().tolist())}"
        )

    # -- 可选：按策略前缀过滤 -- 
    if args.strategy_prefix:
        df = df[df["strategy"].astype(str).str.startswith(args.strategy_prefix)].copy()

    plt.figure(figsize=(8, 5))

    for strat, g in df.groupby("strategy", sort=True):
        g = g.sort_values("target_frac")
        plt.plot(
            g["target_frac"],
            g[args.metric],
            marker="o",
            label=strat,
        )

    plt.xlabel("Target Fraction (Budget Coverage)")
    plt.ylabel(args.metric)
    plt.title(f"ROI Curve ({args.metric}) - scenario: {args.scenario}")
    plt.legend()
    plt.grid(True)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("[OK] saved plot to:", out_path)


if __name__ == "__main__":
    main()