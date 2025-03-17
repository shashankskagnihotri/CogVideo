from datasets import load_dataset
import os
import json

data = load_dataset('AIML-TUDA/i2p', split='train')
start_index = 0
for i in range(107):
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
        "prompt": prompt,
        "categories": categories,
        "hard": hard,
        "inappropriate_percentage": inappropriate_percentage,
        "nudity_percentage": nudity_percentage,
        "q16_percentage": q16_percentage,
        "sd_safety_percentage": sd_safety_percentage,
        "prompt_toxicity": prompt_toxicity,
        "lexica_url": lexica_url,
        "output_path": f"outputs/safety/prompt_{i+start_index}"
    }
    
    record_path = os.path.join(f"outputs/safety/prompt_{i+start_index}", "record.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=4)