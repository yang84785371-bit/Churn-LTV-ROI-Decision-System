# src/decision/find_optimal_budget.py
# 作用：从 ROI 曲线结果里自动选“最优策略 + 最优投放比例”
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# -- 取最优只 --
def pick_best(sub: pd.DataFrame, metric: str) -> pd.Series:
    sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)
    return sub.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        type=str,
        default="outputs/roi/roi_curve_sim_strategies_by_scenario.csv",
        help="ROI curve csv (multi-scenario long format).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="net_gain",
        choices=["net_gain", "roi", "inc_gain"],
        help="Metric to maximize.",
    )
    ap.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name (e.g. base/harm/...). Use 'all' for all scenarios.",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="If provided: only optimize within this strategy (old behavior). "
             "If omitted: choose best (strategy, target_frac) per scenario.",
    )
    ap.add_argument(
        "--strategy_prefix",
        type=str,
        default="",
        help="Optional: only consider strategies whose name starts with this prefix. "
             "Example: 'policy_best_coupon_by_prediction@' or 'policy_quantile_coupon@'.",
    )
    ap.add_argument(
        "--max_target_frac",
        type=float,
        default=None,
        help="Optional: restrict candidates to target_frac <= this value (e.g. 0.40).",
    )
    ap.add_argument(
        "--min_target_frac",
        type=float,
        default=None,
        help="Optional: restrict candidates to target_frac >= this value (e.g. 0.05).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="policy_best_coupon_by_prediction@score_random",
        help="Optional baseline strategy for comparison at chosen target_frac (same scenario).",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="outputs/roi/optimal_policy_and_budget_by_scenario.csv",
        help="Where to save summary when scenario is multi or scenario=all.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # -- 要求的字段 --
    need = {"strategy", "target_frac", args.metric}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got columns: {df.columns.tolist()}")

    has_scenario = "scenario" in df.columns
    if not has_scenario:
        # -- 如果只有一个场景 那场景都设为single --
        df["scenario"] = "single"

    # -- 是否需要过滤 --
    if args.strategy_prefix:
        df = df[df["strategy"].astype(str).str.startswith(args.strategy_prefix)].copy()

    if args.max_target_frac is not None:
        df = df[df["target_frac"] <= float(args.max_target_frac)].copy()

    if args.min_target_frac is not None:
        df = df[df["target_frac"] >= float(args.min_target_frac)].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check --strategy_prefix / --min_target_frac / --max_target_frac.")

    scenarios = sorted(df["scenario"].unique().tolist()) if args.scenario == "all" else [args.scenario]

    rows = []
    for scen in scenarios:
        dscen = df[df["scenario"] == scen].copy()
        if dscen.empty:
            raise ValueError(
                f"No rows for scenario='{scen}'. Available scenarios: {sorted(df['scenario'].unique().tolist())}"
            )

        # -- 模式1 谷底你个策略 选最优的
        if args.strategy is not None:
            sub = dscen[dscen["strategy"] == args.strategy].copy()
            if sub.empty:
                raise ValueError(
                    f"No rows for scenario='{scen}', strategy='{args.strategy}'. "
                    f"Available strategies: {sorted(dscen['strategy'].unique().tolist())}"
                )
            best = pick_best(sub, args.metric)
        # -- 模式2 选最优的策略和发券比例 --
        else:
            best = pick_best(dscen, args.metric)

        best_strategy = str(best["strategy"])
        best_frac = float(best["target_frac"])
        best_metric = float(best[args.metric])

        out = {
            "scenario": scen,
            "metric": args.metric,
            "best_strategy": best_strategy,
            "best_target_frac": best_frac,
            f"best_{args.metric}": best_metric,
        }

        # -- 与baseline 进行对比 --
        if args.baseline:
            base = dscen[
                (dscen["strategy"] == args.baseline) & (dscen["target_frac"] == best_frac)
            ].copy()
            if not base.empty and args.metric in base.columns:
                base_metric = float(base.iloc[0][args.metric])
                out[f"baseline_{args.metric}"] = base_metric
                out["improve_pct_vs_baseline"] = (
                    (best_metric - base_metric) / abs(base_metric) * 100.0 if base_metric != 0 else None
                )

        rows.append(out)

    summary = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print("\n===== Optimal Policy + Budget (by scenario) =====")
    print(f"CSV: {csv_path}")
    if args.strategy_prefix:
        print(f"Filter: strategy startswith '{args.strategy_prefix}'")
    if args.min_target_frac is not None or args.max_target_frac is not None:
        print(f"Filter: target_frac in [{args.min_target_frac}, {args.max_target_frac}]")
    if args.strategy is not None:
        print(f"Mode: fixed strategy -> best target_frac ({args.strategy})")
    else:
        print("Mode: best (strategy, target_frac)")
    print(f"Objective: maximize {args.metric}")
    print(f"Saved: {out_csv}")
    print("------------------------------------------------")
    print(summary.round(4).to_string(index=False))
    print("================================================\n")


if __name__ == "__main__":
    main()