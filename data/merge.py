import glob
import json
import sys
def load_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())
    return data

pattern = sys.argv[1].replace('rank_0','rank_*')
           
files = glob.glob(pattern)
print(files)
input("Continue?")
all_data = []
for fp in files:
    all_data.extend(load_json(fp))
print(len(all_data))
with open(pattern.replace('*','all'),'w') as f:
    f.write(json.dumps(all_data))
    
    
   