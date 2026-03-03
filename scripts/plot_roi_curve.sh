# 画 base 场景
python src/decision/plot_roi_curve.py --scenario base
# 画 harm 场景
python src/decision/plot_roi_curve.py --scenario harm
# 只画 best_coupon 类策略
python src/decision/plot_roi_curve.py \
  --scenario harm \
  --strategy_prefix "policy_best_coupon_by_prediction@"
# 画 ROI 而不是 net_gain
python src/decision/plot_roi_curve.py \
  --scenario base \
  --metric roi