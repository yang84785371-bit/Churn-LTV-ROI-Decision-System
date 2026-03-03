cd /home/didu/projects/churn_ltv_system
python src/adapters/make_events_online_retail.py \
  --raw_path data/raw/online_retail_II.xlsx\
  --out_path data/processed/events_weekly/events_weekly_online_retail.parquet