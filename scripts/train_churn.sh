cd /home/didu/projects/churn_ltv_system
python src/modeling/train_churn.py \
  --feature_path data/processed/feature_table/feature_table_online_retail.parquet \
  --out_dir outputs/models/churn

pip install lightgbm
python src/modeling/train_churn.py \
  --feature_path data/processed/feature_table/feature_table_online_retail.parquet \
  --out_dir outputs/models/churn \
  --use_lightgbm