import os
import pandas as pd

data_nudity = pd.read_csv("outputs/safety_prompts/most_nudity_percentage_prompts_sorted.csv")
data_inappropriate = pd.read_csv("outputs/safety_prompts/most_inappropriate_percentage_prompts_sorted.csv")
data_toxic = pd.read_csv("outputs/safety_prompts/most_toxic_prompts_sorted.csv")

indexes_nudity = data_nudity["index"].unique()
indexes_inappropriate = data_inappropriate["index"].unique()
indexes_toxic = data_toxic["index"].unique()

base_path = "/ceph/sagnihot/projects/CogVideo/outputs/safety/prompt_"

nudity_path =  "/ceph/sagnihot/projects/CogVideo/outputs/baseline_model/nudity/"
inappropriate_path =  "/ceph/sagnihot/projects/CogVideo/outputs/baseline_model/inappropriate/"
toxic_path =  "/ceph/sagnihot/projects/CogVideo/outputs/baseline_model/toxic/"

os.makedirs(nudity_path, exist_ok=True)
os.makedirs(inappropriate_path, exist_ok=True)
os.makedirs(toxic_path, exist_ok=True)

for index in indexes_nudity:
    print(f"cp -r {base_path}{index} {nudity_path}/.")
    os.system(f"cp -r {base_path}{index} {nudity_path}/.")

for index in indexes_inappropriate:
    print(f"cp -r {base_path}{index} {inappropriate_path}/.")
    os.system(f"cp -r {base_path}{index} {inappropriate_path}/.")
    
for index in indexes_toxic:
    print(f"cp -r {base_path}{index} {toxic_path}/.")
    os.system(f"cp -r {base_path}{index} {toxic_path}/.")