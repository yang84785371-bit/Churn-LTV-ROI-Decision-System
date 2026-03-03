cd /home/didu/projects/churn_ltv_system
python src/modeling/train_ltv.py \
  --feature_path data/processed/feature_table/feature_table_online_retail.parquet \
  --out_dir outputs/models/ltv \
  --use_lightgbm