import pandas as pd
data = pd.read_json('/scratch/users/zenkavi/tmp/data/mturk_validation_data_post.json')
completion_dates = data.groupby(['worker_id'], sort=False)['finishtime'].max()
completion_dates.to_csv('/scratch/users/zenkavi/tmp/data/validation_completion_dates.csv')
