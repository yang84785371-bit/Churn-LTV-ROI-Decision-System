cd /home/didu/projects/churn_ltv_system
python src/features/build_features.py \
  --events_path data/processed/events_weekly/events_weekly_online_retail.parquet \
  --out_path data/processed/feature_table/feature_table_online_retail.parquet