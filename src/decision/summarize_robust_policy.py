# src/decision/summarize_robust_policy.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_path", type=str, default="outputs/roi/roi_curve_sim_strategies_by_scenario.csv")
    ap.add_argument("--metric", type=str, default="net_gain", choices=["net_gain", "roi", "inc_gain"])
    ap.add_argument("--strategy_prefix", type=str, default="", help="optional: filter strategies by prefix")
    ap.add_argument("--out_csv", type=str, default="outputs/roi/robust_policy_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.roi_path)
    if "scenario" not in df.columns:
        raise ValueError("roi_path must contain 'scenario' column (multi-scenario long format).")
    if args.metric not in df.columns:
        raise ValueError(f"metric '{args.metric}' not in columns: {df.columns.tolist()}")

    if args.strategy_prefix:
        df = df[df["strategy"].astype(str).str.startswith(args.strategy_prefix)].copy()
        if df.empty:
            raise ValueError("No rows after strategy_prefix filtering.")

    # -- 对于每一个 (scenario, strategy), 选出某标准下的最优 --
    best_by_scen = (
        df.sort_values(args.metric, ascending=False)
          .groupby(["scenario", "strategy"], as_index=False)
          .first()
          .rename(columns={"target_frac": "best_target_frac"})
    )

    # -- 对不同场景进行聚合来展示稳健性 --
    agg = (
        best_by_scen.groupby("strategy")
        .agg(
            scenarios=("scenario", "nunique"),
            mean_metric=(args.metric, "mean"),
            worst_case_metric=(args.metric, "min"),
            best_frac_median=("best_target_frac", "median"),
            best_frac_mean=("best_target_frac", "mean"),
        )
        .reset_index()
        .sort_values(["worst_case_metric", "mean_metric"], ascending=False)
    )

    # -- 挑选最稳健的策略 --
    winner = agg.iloc[0].to_dict()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)

    print("\n===== Robust Policy Summary =====")
    print(f"Input: {args.roi_path}")
    print(f"Metric: {args.metric}")
    if args.strategy_prefix:
        print(f"Filter: strategy startswith '{args.strategy_prefix}'")
    print(f"Saved: {out_csv}")
    print("--------------------------------")
    print(agg.round(4).to_string(index=False))
    print("--------------------------------")
    print("[RECOMMEND]")
    print(f"robust_best_strategy: {winner['strategy']}")
    print(f"robust_budget_frac (median of per-scenario optima): {winner['best_frac_median']:.2f}")
    print(f"worst_case_{args.metric}: {winner['worst_case_metric']:.2f}")
    print("================================\n")


if __name__ == "__main__":
    main()