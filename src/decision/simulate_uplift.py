'''
    这里我们通过在不同的场景下
    模拟出uplift（挽回用户可获得的收益）
    来得到不同的券档位下可以有的成本和收益
    从而选出最优秀的发券策略
    给每个用户、每种券，在每种场景下，生成一个“真实收益表”。
'''

# src/decision/simulate_uplift.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import zlib

def main():
    # -- 命令行参数 --
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_path", type=str, default="outputs/models/ltv/scored_test_ltv.parquet")
    ap.add_argument("--churn_path", type=str, default="outputs/models/churn/calibrated/scored_test.parquet")
    ap.add_argument("--out_path", type=str, default="outputs/roi/simulated_test.parquet")
    ap.add_argument("--base_uplift", type=float, default=0.30)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--mode", type=str, default="risk_value", choices=["risk", "value", "risk_value"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", type=str, default="base,misalign,saturation,high_noise,harm",
                help="comma-separated scenario names")
    ap.add_argument("--out_path_all", type=str, default=None,
                    help="optional: save all scenarios to another parquet path")
    args = ap.parse_args()
    # -- 读取ltv 和 churn的数据 --
    ltv = pd.read_parquet(args.scored_path)
    churn = pd.read_parquet(args.churn_path)

    # -- 避免 _x/_y：只保留需要的列 --
    churn_keep = churn[["user_id", "cutoff_week", "p_churn_base", "p_churn_cal"]].copy()
    ltv_keep = ltv[["user_id", "cutoff_week", "future_margin_4w", "expected_margin_4w"]].copy()

    df = churn_keep.merge(ltv_keep, on=["user_id", "cutoff_week"], how="inner")

    df["p_churn"] = df["p_churn_cal"].astype(float) # 流失率用校正过的

    # -- 构造uplift 可以用预测值 不能用真值 --
    value_rank = df["expected_margin_4w"].rank(pct=True).astype(float)  # 0..1
    df["value_rank"] = value_rank

    '''
        多场景配置
    '''
    scenario_cfg = {
        # -- 1) base: 你现在的默认世界--
        "base": dict(mode=args.mode, base_uplift=args.base_uplift, noise_std=args.noise_std,
                    misalign="none", sat_alpha=0.0, harm_prob=0.0, harm_scale=0.0),

        # -- 2) misalign: uplift 与 score_joint 错配 高风险高价值不一定更容易被救 --
        #    思路：把 interaction 反过来，让value churn 同时高不占便宜
        "misalign": dict(mode="risk_value", base_uplift=args.base_uplift, noise_std=args.noise_std,
                        misalign="anti_joint", sat_alpha=0.0, harm_prob=0.0, harm_scale=0.0),

        # -- 3) saturation: 饱和/渠道疲劳越往长尾投放，平均 uplift 越低--
        "saturation": dict(mode="risk_value", base_uplift=args.base_uplift, noise_std=args.noise_std,
                        misalign="none", sat_alpha=0.35, harm_prob=0.0, harm_scale=0.0),

        # -- 4) high_noise: 环境更不稳定 同样策略方差更大 --
        "high_noise": dict(mode="risk_value", base_uplift=args.base_uplift, noise_std=max(args.noise_std, 0.15),
                        misalign="none", sat_alpha=0.0, harm_prob=0.0, harm_scale=0.0),

        # -- 5) harm: 负效应/挤出部分人发券反而亏：退货/薅羊毛/本来就会买 --
        "harm": dict(mode="risk_value", base_uplift=args.base_uplift, noise_std=args.noise_std,
                    misalign="none", sat_alpha=0.0, harm_prob=0.20, harm_scale=0.60),
    }

    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenario_names = [s for s in scenario_names if s in scenario_cfg]
    if not scenario_names:
        scenario_names = ["base"]

    '''
        生成多重收益世界 
    '''
    base_rows = [] # 容器
    p = df["p_churn"].astype(float).to_numpy()                 # 流失率
    v = df["value_rank"].astype(float).to_numpy()              # 预测值的rank
    for scen in scenario_names:
        cfg = scenario_cfg[scen] # 对应世界的配置
        scenario_seed = args.seed + (zlib.crc32(scen.encode()) % 100000) # 随机种子 稳定 但不同
        rng_s = np.random.default_rng(scenario_seed) # 创建一个随机数生成器。

        # --  非线性变换 -- 
        p_nl = np.sqrt(np.clip(p, 0, 1)) # 高值变得没那么极端 低值相对被放大一点
        v_nl = np.sqrt(np.clip(v, 0, 1))

        # -- 交互项 -- 
        interaction = np.clip(p_nl + v_nl - 1.0, 0.0, 1.0)   #风险高和价值高同时成立时，interaction 才会大。

        # -- 错配开关 -- 
        # -- 错配世界：同时高 -> 反而不占优势 -- 
        if cfg["misalign"] == "anti_joint":
            interaction_used = 1.0 - interaction
        else:
            interaction_used = interaction

        # -- 决定 uplift 强弱的主因子 --
        if cfg["mode"] == "risk":
            driver = 0.85 * p_nl + 0.15 * rng_s.random(len(p))
        elif cfg["mode"] == "value":
            driver = 0.85 * v_nl + 0.15 * rng_s.random(len(p))
        else:  # risk_value
            driver = 0.35 * p_nl + 0.20 * v_nl + 0.45 * interaction_used

        # -- 在logit 空间加噪声 让 uplift 强度更真实、有随机性，而不是完全规则化 --
        eps = rng_s.normal(0, cfg["noise_std"], size=len(p)) # 扰动项
        logit = (driver - 0.5) * 2.0 + eps
        sigmoid = 1.0 / (1.0 + np.exp(-logit))
        base_intensity = sigmoid  # 0..1

        # --  饱和效应  --
        # -- sat_alpha > 0：让长尾即低 value_rank更难 uplift，放大边际递减 --
        sat_alpha = float(cfg["sat_alpha"])
        if sat_alpha > 0:
            base_intensity = np.clip(base_intensity * (0.65 + 0.35 * v) * (1.0 - sat_alpha * (1.0 - v)), 0.0, 1.0)

        # -- 这个世界整体营销效果强度 * 每个人“容易被救”的强度 --
        true_uplift = np.clip(float(cfg["base_uplift"]) * base_intensity, 0.0, 1.0)

        # -- 多档券设计 --
        coupon_costs = np.array([5.0, 15.0, 25.0]) # 发券成本
        uplift_multipliers = np.array([1.0, 1.6, 2.2]) # 券的效果强度倍数

        # -- 再加一层随机扰动 即便两个用户强度一样，现实效果也会有波动。
        eps_u = rng_s.normal(0, cfg["noise_std"], size=len(df))
        base_intensity_noisy = np.clip(base_intensity + eps_u, 0.0, 1.0)
        # -- 构造三档券的 uplift 矩阵 -- 
        uplift_matrix = np.outer(float(cfg["base_uplift"]) * base_intensity_noisy, uplift_multipliers)
        uplift_matrix = np.clip(uplift_matrix, 0.0, 1.0)
        # -- 真实的未来收益 --
        future_margin = df["future_margin_4w"].to_numpy().reshape(-1, 1)
        cost_matrix = coupon_costs.reshape(1, -1) # 券的成本

        # -- harm --
        # harm_prob：一定概率出现负效应；harm_scale：负效应强度（按未来利润比例扣减）
        harm_prob = float(cfg["harm_prob"]) #读取场景参数
        harm_scale = float(cfg["harm_scale"])
        harm_mask = rng_s.random(len(df)) < harm_prob # 随机抽一批会出问题的用户
        harm_penalty = harm_scale * np.clip(future_margin, 0.0, None)  # 给这些用户算一个惩罚金额
        harm_penalty = harm_penalty.reshape(-1)

        gain_matrix = uplift_matrix * future_margin  # 先算三档券的真实增量收益（不含成本）
        if harm_prob > 0:
            # -- 如果用户被抽中 harm，就把三档券的增量收益都扣掉一部分   --
            gain_matrix[harm_mask, :] = np.clip(gain_matrix[harm_mask, :] - harm_penalty[harm_mask].reshape(-1, 1), 0.0, None)

        net_gain_matrix = gain_matrix - cost_matrix # 净收益

        best_choice = net_gain_matrix.argmax(axis=1) # 挑选净收益最高的
        best_net = net_gain_matrix.max(axis=1) # 最佳收益

        no_treat_mask = best_net < 0 # 如果最佳收益小于0 那就不做调整

        d2 = df.copy() # 复制一份原始打分数据
        d2["scenario"] = scen # 标记这一份数据属于哪个场景
        d2["true_uplift"] = true_uplift # uplift 强度
        # -- 三档券增量收益 --
        d2["gain_small"] = gain_matrix[:, 0]
        d2["gain_medium"] = gain_matrix[:, 1]
        d2["gain_large"] = gain_matrix[:, 2]

        d2["best_coupon"] = best_choice
        d2.loc[no_treat_mask, "best_coupon"] = -1 # 意思就是找到 "best_coupon"中no_treat_mask 对应为1的部分赋予-1

        d2["treatment_cost"] = coupon_costs[best_choice]
        d2.loc[no_treat_mask, "treatment_cost"] = 0.0
        # -- 用户最优发券策略的收益 --
        d2["true_incremental_gain"] = np.where(
            no_treat_mask,
            0.0,
            gain_matrix[np.arange(len(d2)), best_choice]
        )

        base_rows.append(d2)

    all_df = pd.concat(base_rows, ignore_index=True)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(args.out_path, index=False)
    print("[OK] Simulated uplift data (multi-scenario) saved to:", args.out_path)

    if args.out_path_all:
        Path(args.out_path_all).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_parquet(args.out_path_all, index=False)
        print("[OK] also saved:", args.out_path_all)

    print(all_df["scenario"].value_counts())

    print("[INFO] mode:", args.mode)


if __name__ == "__main__":
    main()