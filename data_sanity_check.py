import sys
from transformers import AutoTokenizer
file_name = sys.argv[1]
model = sys.argv[2]
import json

with open(file_name) as f:
    data = json.loads(f.read())
data = data[0]
tokenizer = AutoTokenizer.from_pretrained(model)
z = tokenizer.batch_decode([data['complete_teacher_output_ids']])
breakpoint()