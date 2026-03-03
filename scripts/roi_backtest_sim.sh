cd /home/didu/projects/churn_ltv_system
python src/decision/roi_backtest_sim.py \
  --sim_path outputs/roi/simulated_test.parquet \
  --out_dir outputs/roi \
  --touch_cost 10

  python src/decision/roi_backtest_sim.py \
  --sim_path outputs/roi/sim_multi_coupon.parquet \
  --out_dir outputs/roi

python src/decision/roi_backtest_sim.py \
  --sim_path outputs/roi/simulated_test.parquet \
  --out_dir outputs/roi \
  --seed 42