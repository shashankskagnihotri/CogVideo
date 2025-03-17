from datasets import load_dataset
import os
import json
import pandas as pd

data = load_dataset('AIML-TUDA/i2p', split='train')
start_index = 0
all_prompts = []

for i in range(len(data)):
    prompt = data[i]['prompt']
    categories = data[i]['categories']
    hard = data[i]['hard']
    inappropriate_percentage = data[i]['inappropriate_percentage']
    nudity_percentage = data[i]['nudity_percentage']
    q16_percentage = data[i]['q16_percentage']
    sd_safety_percentage = data[i]['sd_safety_percentage']
    prompt_toxicity = data[i]['prompt_toxicity']
    lexica_url = data[i]['lexica_url']
    
    record = {
        "index": i,
        "prompt": prompt,
        "categories": categories,
        "hard": hard,
        "inappropriate_percentage": inappropriate_percentage,
        "nudity_percentage": nudity_percentage,
        "q16_percentage": q16_percentage,
        "sd_safety_percentage": sd_safety_percentage,
        "prompt_toxicity": prompt_toxicity,
        "lexica_url": lexica_url,
    }
    
    all_prompts.append(record)

df = pd.DataFrame(all_prompts)
os.makedirs('outputs/safety_prompts', exist_ok=True)
#import ipdb; ipdb.set_trace()

df.sort_values(by='prompt_toxicity', ascending=False, inplace=False).to_csv('outputs/safety_prompts/prompts_sorted_by_toxicity.csv', index=False)
df.sort_values(by='prompt_toxicity', ascending=False, inplace=False)[:10].to_csv('outputs/safety_prompts/most_toxic_prompts_sorted.csv', index=False)

df.sort_values(by='nudity_percentage', ascending=False, inplace=False).to_csv('outputs/safety_prompts/prompts_sorted_by_nudity_percentage.csv', index=False)
df.sort_values(by='nudity_percentage', ascending=False, inplace=False)[:10].to_csv('outputs/safety_prompts/most_nudity_percentage_prompts_sorted.csv', index=False)

df.sort_values(by='inappropriate_percentage', ascending=False, inplace=False).to_csv('outputs/safety_prompts/prompts_sorted_by_inappropriate_percentage.csv', index=False)
df.sort_values(by='inappropriate_percentage', ascending=False, inplace=False)[:10].to_csv('outputs/safety_prompts/most_inappropriate_percentage_prompts_sorted.csv', index=False)