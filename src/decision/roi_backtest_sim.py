# src/decision/roi_backtest_sim.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# -- 确保字段 --
def _ensure_cols(df: pd.DataFrame, need: List[str]) -> None:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Got columns: {df.columns.tolist()}"
        )

# -- 默认的用户的分位数 -- 
def _default_fracs() -> np.ndarray:
    return np.linspace(0.05, 0.50, 10)

# -- 得到分位数 --
def _parse_fracs(fracs_str: str | None) -> np.ndarray:
    if not fracs_str:
        return _default_fracs()
    # e.g. "0.05,0.1,0.2"
    xs = [float(x.strip()) for x in fracs_str.split(",") if x.strip()]
    xs = [x for x in xs if 0 < x <= 1]
    if not xs:
        return _default_fracs()
    return np.array(xs, dtype=float)

# -- 计算四种得分 -- 
def _make_scores(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    d = df.copy()
    # 统一口径的排序分数
    d["score_churn"] = d["p_churn_cal"].astype(float)
    d["score_value"] = d["expected_margin_4w"].astype(float)
    d["score_joint"] = d["p_churn_cal"].astype(float) * d["expected_margin_4w"].astype(float)
    rng = np.random.default_rng(seed)
    d["score_random"] = rng.random(len(d))
    return d


"""
    在 top-k 内按固定比例发券：
    - 前 share_large 发大券
    - 接着 share_medium 发中券
    - 接着 share_small 发小券
    - 剩余不发券

    注意：这是一种“可执行规则”，完全不依赖 simulate_uplift 里 oracle 的 best_coupon/treatment_cost。
"""
def roi_curve_policy_quantile_coupon(
    df: pd.DataFrame,
    base_score_col: str,
    fracs: np.ndarray,
    coupon_costs: np.ndarray,
    share_large: float = 0.05,
    share_medium: float = 0.10,
    share_small: float = 0.15,
) -> pd.DataFrame:
    d = df.sort_values(base_score_col, ascending=False).reset_index(drop=True) # 排序 按照高分的先选
    N = len(d)# 总人数
    rows: List[Dict] = [] # 容器

    for frac in fracs: # 确定发券人数的百分比
        k = max(1, int(N * float(frac))) # 确定发券的总人数
        sub = d.iloc[:k].copy() # 筛选出发券的用户

        m = len(sub) # 子集人数
        n_large = int(round(m * share_large)) # 分别是大券人数 中券人数 小券人数
        n_med = int(round(m * share_medium))
        n_small = int(round(m * share_small))

        # -- 防止发券人数超过用户数越界 -- 
        n_large = min(n_large, m)
        n_med = min(n_med, m - n_large)
        n_small = min(n_small, m - n_large - n_med)

        # 真实收益（来自 simulate_uplift 生成的三档真实增量收益）
        gain_large = sub["gain_large"].to_numpy(dtype=float)
        gain_med = sub["gain_medium"].to_numpy(dtype=float)
        gain_small = sub["gain_small"].to_numpy(dtype=float)

        total_gain = 0.0
        total_cost = 0.0

        treat_cnt = 0
        cnt_small = 0
        cnt_med = 0
        cnt_large = 0

        # 大券算收益
        if n_large > 0:
            total_gain += float(gain_large[:n_large].sum()) # 总的净收益
            total_cost += float(coupon_costs[2] * n_large) # 总的成本
            treat_cnt += n_large # 逐步记录总的发圈人数
            cnt_large += n_large # 大券人数

        # 中券
        l2, r2 = n_large, n_large + n_med
        if r2 > l2:
            total_gain += float(gain_med[l2:r2].sum())
            total_cost += float(coupon_costs[1] * (r2 - l2))
            treat_cnt += (r2 - l2)
            cnt_med += (r2 - l2)

        # 小券
        l3, r3 = n_large + n_med, n_large + n_med + n_small
        if r3 > l3:
            total_gain += float(gain_small[l3:r3].sum())
            total_cost += float(coupon_costs[0] * (r3 - l3))
            treat_cnt += (r3 - l3)
            cnt_small += (r3 - l3)

        no_treat_cnt = k - treat_cnt

        rows.append({
            "strategy": f"policy_quantile_coupon@{base_score_col}",
            "target_frac": float(frac),
            "num_users": int(k),
            "inc_gain": total_gain,
            "cost": total_cost,
            "net_gain": total_gain - total_cost,
            "roi": (total_gain - total_cost) / total_cost if total_cost > 0 else np.nan,
            "treat_cnt": int(treat_cnt),
            "no_treat_cnt": int(no_treat_cnt),
            "share_treated": (treat_cnt / k) if k > 0 else np.nan,
            "share_small": (cnt_small / treat_cnt) if treat_cnt > 0 else np.nan,
            "share_medium": (cnt_med / treat_cnt) if treat_cnt > 0 else np.nan,
            "share_large": (cnt_large / treat_cnt) if treat_cnt > 0 else np.nan,
            "avg_cost_per_treated": (total_cost / treat_cnt) if treat_cnt > 0 else np.nan,
        })

    return pd.DataFrame(rows)


'''
    先用某个 base score 选人，再在被选中的人里，用预测净收益最大化选最优券。
'''
def roi_curve_policy_best_coupon(
    df: pd.DataFrame,
    base_score_col: str,
    fracs: np.ndarray,
    coupon_costs: np.ndarray,
    uplift_multipliers: np.ndarray,
    beta: float = 0.5,
    scale: float = 4.0,
    cap_large: float = 0.2,
    min_pred_net: float = 0.0
) -> pd.DataFrame:
    d = df.sort_values(base_score_col, ascending=False).reset_index(drop=True) # 先排序
    N = len(d) # 总的用户数
    rows: List[Dict] = [] # 容器

    # -- churn probability --
    p = d["p_churn_cal"].astype(float).to_numpy()
    v_rank = d["expected_margin_4w"].rank(pct=True).astype(float).to_numpy()  # value rank 
    uplift_proxy = scale * p * (v_rank ** beta) # 计算uplift的proxy

    v_pred = d["expected_margin_4w"].astype(float).to_numpy()  # 用预测价值做收益基底
    pred_gain_matrix = np.outer(uplift_proxy, uplift_multipliers) * v_pred.reshape(-1, 1) # 计算预测收益
    pred_net_matrix = pred_gain_matrix - coupon_costs.reshape(1, -1) # 计算预测净利润

    best_coupon_all = pred_net_matrix.argmax(axis=1)   # 0/1/2 选利润最大的做为
    best_net_all = pred_net_matrix.max(axis=1)         # max net

    gain_cols = ["gain_small", "gain_medium", "gain_large"] # 字段给个名 

    for frac in fracs:# 用户分位数
        k = max(1, int(N * float(frac))) #选中的用户人数
        sub = d.iloc[:k].copy() # 用个子表将信息选出来

        # -- top-k 局部决策（不污染全局） --
        best_coupon_k = best_coupon_all[:k].copy() # 根据proxy 选出最适合的coupon
        best_net_k = best_net_all[:k].copy() # 最优的净收益 
        pred_net_k = pred_net_matrix[:k, :].copy() # 以及相对应的预测收益网络

        # -- 约束 --
        max_large = int(np.floor(k * cap_large))
        if max_large < 0:
            max_large = 0

        if max_large < k:
            # -- 大券相对中券的收益优势 --
            large_adv = pred_net_k[:, 2] - pred_net_k[:, 1]
            rank = np.argsort(-large_adv)  # desc 
            allowed_large = set(rank[:max_large])

            for i in range(k):
                if best_coupon_k[i] == 2 and i not in allowed_large: # 这里限制大券数量了 只能重新选小券或者中券
                    # 禁用大券，在 small/medium 中重选
                    alt = int(np.argmax(pred_net_k[i, :2]))
                    best_coupon_k[i] = alt
                    best_net_k[i] = float(np.max(pred_net_k[i, :2]))

        # -- 发券：预测净收益 < 0 --
        no_treat_k = best_net_k < min_pred_net

        # -- 用 sub 的真实 gain 结算（索引对齐 sub）--
        sub_gain = sub[gain_cols].to_numpy(dtype=float)
        # --- 这里进行结算 ---
        total_gain = 0.0  # 总收益
        total_cost = 0.0 # 总成本
        treat_cnt = 0 # 累计的发券数量
        no_treat_cnt = 0 # 累计的因为收益负数不发券的人数
        cnt_small = 0 # 发小
        cnt_med = 0 # 发中
        cnt_large = 0 # 发大
        # 下面就是逐个用户进行计算的工程
        for i in range(k):
            if no_treat_k[i]:
                no_treat_cnt += 1
                continue

            c = int(best_coupon_k[i])  # 0/1/2
            treat_cnt += 1

            if c == 0:
                cnt_small += 1
            elif c == 1:
                cnt_med += 1
            else:
                cnt_large += 1

            total_gain += float(sub_gain[i, c])
            total_cost += float(coupon_costs[c])

        rows.append({
            "strategy": f"policy_best_coupon_by_prediction@{base_score_col}",
            "target_frac": float(frac),
            "num_users": int(k),
            "inc_gain": total_gain,
            "cost": total_cost,
            "net_gain": total_gain - total_cost,
            "roi": (total_gain - total_cost) / total_cost if total_cost > 0 else np.nan,
            "treat_cnt": int(treat_cnt),
            "no_treat_cnt": int(no_treat_cnt),
            "share_treated": (treat_cnt / k) if k > 0 else np.nan,
            "share_small": (cnt_small / treat_cnt) if treat_cnt > 0 else np.nan,
            "share_medium": (cnt_med / treat_cnt) if treat_cnt > 0 else np.nan,
            "share_large": (cnt_large / treat_cnt) if treat_cnt > 0 else np.nan,
            "avg_cost_per_treated": (total_cost / treat_cnt) if treat_cnt > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ----------------------------
# main
# ----------------------------
def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_path", type=str, default="outputs/roi/simulated_test.parquet")
    ap.add_argument("--out_dir", type=str, default="outputs/roi")

    #  -- 用户切分 --
    ap.add_argument("--fracs", type=str, default=None, help='e.g. "0.05,0.1,0.2,0.5"')

    # -- 券的设计 包括成本与uplift效果 -- 
    ap.add_argument("--coupon_costs", type=str, default="5,15,25", help='e.g. "5,15,25"')
    ap.add_argument("--uplift_multipliers", type=str, default="1.0,1.6,2.2", help='e.g. "1.0,1.6,2.2"')

    # -- coupon proxy的超参 --
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--cap_large", type=float, default=0.2)

    # -- 策略共享的分位数 --
    ap.add_argument("--q_large", type=float, default=0.05)
    ap.add_argument("--q_medium", type=float, default=0.10)
    ap.add_argument("--q_small", type=float, default=0.15)
    # -- 随机种子 --
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_pred_net", type=float, default=0.0,
                help="minimum predicted net gain required to issue coupon") # 发券最低的预测净收益
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # -- 读取数据 --
    df = pd.read_parquet(args.sim_path)
    # -- 要求的字段列表 --
    need = [
        "user_id", "cutoff_week",
        "p_churn_cal", "expected_margin_4w",
        "gain_small", "gain_medium", "gain_large",
        "scenario",
    ]
    # -- 确保字段都有 -- 
    _ensure_cols(df, need)
    # -- 得到fracs用于后续用户切分 --
    fracs = _parse_fracs(args.fracs)

    # -- 都读参数 然后防御一下 --
    coupon_costs = np.array([float(x) for x in args.coupon_costs.split(",")], dtype=float)
    if coupon_costs.shape[0] != 3:
        raise ValueError("--coupon_costs must have 3 values, e.g. 5,15,25")

    uplift_multipliers = np.array([float(x) for x in args.uplift_multipliers.split(",")], dtype=float)
    if uplift_multipliers.shape[0] != 3:
        raise ValueError("--uplift_multipliers must have 3 values, e.g. 1.0,1.6,2.2")
    
    '''
        计算得分 将标准分为 random joint value 和 churn
    '''
    d = _make_scores(df, seed=args.seed)
    # -- 容器 -- 
    all_curves: List[pd.DataFrame] = []
    # -- 得分字段 --
    score_cols = ["score_random", "score_churn", "score_value", "score_joint"]

    # -- 对每一种场景进行run -- 
    for scen, sdf in d.groupby("scenario", sort=True): # groupby之后分成不同的场景小表
        for col in score_cols: # 然后逐个标准进行
            c1 = roi_curve_policy_quantile_coupon(
                sdf, base_score_col=col, fracs=fracs, coupon_costs=coupon_costs,
                share_large=args.q_large, share_medium=args.q_medium, share_small=args.q_small
            ) # 根据quantile策略进行发券
            c1["scenario"] = scen # 备注上scen
            all_curves.append(c1) # 记录
            # -- 同理 这里是个人化的best coupon --
            c2 = roi_curve_policy_best_coupon(
                sdf, base_score_col=col, fracs=fracs, coupon_costs=coupon_costs,
                uplift_multipliers=uplift_multipliers,
                beta=args.beta, scale=args.scale, cap_large=args.cap_large, min_pred_net=args.min_pred_net
            )
            c2["scenario"] = scen
            all_curves.append(c2)
    # -- 全部cat到一起 去掉重复的表头 --
    curves = pd.concat(all_curves, ignore_index=True)
    # -- 进行保存
    out_csv = out_dir / "roi_curve_sim_strategies_by_scenario.csv"
    curves.to_csv(out_csv, index=False)

    print("[OK] saved:", out_csv)

    print("[OK] saved:", out_csv)

    for scen, g in curves.groupby("scenario", sort=True):
        pv = g.pivot_table(index="target_frac", columns="strategy", values="net_gain") # 打印不同场景下，各策略的净收益表
        pv.round(2).to_csv(out_dir / f"net_gain_pivot_{scen}.csv")

    best_mask = curves["strategy"].str.startswith("policy_best_coupon_by_prediction@")
    best_df = curves.loc[best_mask].copy()
    if not best_df.empty:
        cols = ["scenario", "strategy", "target_frac", "share_treated", "share_small", "share_medium", "share_large", "avg_cost_per_treated"]
        print("\n[Policy breakdown] best_coupon_by_prediction (by base score, per scenario)")
        for scen, g in best_df.groupby("scenario", sort=True): # 打印 best_coupon 策略在不同场景下的发券结构。
            g.to_csv(out_dir / f"best_coupon_breakdown_{scen}.csv", index=False)
        best_df.to_csv(out_dir / "best_coupon_breakdown_all_scenarios.csv", index=False)

if __name__ == "__main__":
    main()