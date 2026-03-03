PYTHON=python

SCORED_PATH=outputs/models/ltv/scored_test_ltv.parquet
CHURN_PATH=outputs/models/churn/calibrated/scored_test.parquet
OUT_PATH=outputs/roi/simulated_test.parquet
OUT_ALL=outputs/roi/simulated_test_all_scenarios.parquet

$PYTHON src/decision/simulate_uplift.py \
  --scored_path $SCORED_PATH \
  --churn_path $CHURN_PATH \
  --out_path $OUT_PATH \
  --out_path_all $OUT_ALL \
  --mode risk_value \
  --base_uplift 0.30 \
  --noise_std 0.08 \
  --scenarios base,misalign,saturation,high_noise,harm \
  --seed 42

echo "[DONE] multi-scenario uplift simulation finished."