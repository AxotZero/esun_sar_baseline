import pandas as pd

folders = []

n_folder = len(folders)
outputs = {}
for folder in folders:
    df = pd.read_csv(f'{folder}/submission.csv')
    for _, row in df.iterrows():
        k, v = row['alert_key'], row['probability']
        if k not in outputs:
            outputs[k] = 0
        outputs[k] += v

# mean
for k, v in outputs.items():
    outputs[k] /= n_folder

submit = pd.DataFrame(
    data={
        'alert_key': list(outputs.keys()), 
        'probability': list(outputs.values())
    }
)
submit['alert_key'] = submit['alert_key'].astype(int)
submit.sort_values(by='probability', inplace=True)
submit.to_csv(f'/media/hd03/axot_data/sar/save_dir/ensemble.csv', index=None)