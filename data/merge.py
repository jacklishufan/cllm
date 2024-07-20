import glob
import json
def load_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())
    return data

pattern = '/home/jacklishufan/Consistency_LLM/data/collected_jacobi_trajectory/cleaned_gsm8k_train_jacobi_max_new_tokens16_augTrue_labels_True_max_seq_len_1024_rank_*.json'
           
files = glob.glob(pattern)
all_data = []
for fp in files:
    all_data.extend(load_json(fp))
print(len(all_data))
with open(pattern.replace('*','all'),'w') as f:
    f.write(json.dumps(all_data))