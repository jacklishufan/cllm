import glob
import json
def load_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())
    return data

pattern = '/home/bootstrap/jacklishufan/Consistency_LLM/data/collected_jacobi_trajectory/cleaned_code_search_net_jacobi_max_new_tokens64_augTrue_labels_True_max_seq_len_1024_rank_*_0.json'
           
files = glob.glob(pattern)
print(files)
all_data = []
for fp in files:
    all_data.extend(load_json(fp))
print(len(all_data))
with open(pattern.replace('*','all'),'w') as f:
    f.write(json.dumps(all_data))