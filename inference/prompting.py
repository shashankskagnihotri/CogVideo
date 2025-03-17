import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号
MODEL_PATH = "THUDM/glm-4-9b-chat-hf"
#MODEL_PATH = "THUDM/glm-4-9b"

device = "cuda" if torch.cuda.is_available() else "cpu"
#import ipdb; ipdb.set_trace()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
#model = AutoModel.from_pretrained("THUDM/glm-4-9b-hf", trust_remote_code=True)

query = "A girl riding a bike."
#'''
inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
#'''
#import ipdb; ipdb.set_trace()
#inputs = tokenizer(query, return_tensors="pt")

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import ipdb; ipdb.set_trace()