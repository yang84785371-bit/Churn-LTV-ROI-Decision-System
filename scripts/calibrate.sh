cd /home/didu/projects/churn_ltv_system
python src/modeling/calibrate.py \
  --feature_path data/processed/feature_table/feature_table_online_retail.parquet \
  --model_path outputs/models/churn/lightgbm.joblib \
  --out_dir outputs/models/churn/calibrated \
  --method isotonic