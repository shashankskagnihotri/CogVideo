import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import spaces
import cv2
import os
import argparse
from PIL import Image
import glob
import json

"""
REQUIRED PACKAGES:
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process video and generate captions.")
parser.add_argument("--video_path", type=str, required=False, default='outputs/testing_ablitearted_glm4_without_prompt_prefix/testing_nudity/using_glm/prompt_2110/output_2110.mp4', help="Path to the input video file.")
args = parser.parse_args()

model_info = 'allenai/Molmo-7B-D-0924'
#model_info = 'allenai/Molmo-7B-O-0924'
#model_info = 'allenai/MolmoE-1B-0924'
processor = AutoProcessor.from_pretrained(
    model_info,
    trust_remote_code=True,
    torch_dtype='bfloat16',
    device_map='cuda'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_info,
    trust_remote_code=True,
    torch_dtype='bfloat16',
    device_map='cuda'
)


def extract_frames(video_path, output_dir, num_frames=5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration = total_frames / fps  # Duration of the video in seconds

    # Calculate frame indices to extract
    frame_indices = [int(fps * i) for i in range(0, min(num_frames, int(duration)))]

    extracted_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the frame position
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"frame_{idx}.jpg")
            cv2.imwrite(frame_path, frame)  # Save the frame as an image
            extracted_frames.append(frame_path)

    cap.release()
    return extracted_frames


video_paths = glob.glob('outputs/testing_ablitearted_glm4_without_prompt_prefix/**/using_glm/**/*.mp4')
for video_path in video_paths:
    prompt_number = video_path.split("prompt_")[-1].split("/")[0]
    prompt_type = video_path.split("testing_")[-1].split("/")[0]
    
    json_path = video_path.replace("output_{}.mp4".format(prompt_number), "record.json")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_dir = "outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_{}/prompt_{}".format(prompt_type, prompt_number)
    # Create the output directory if it doesn't exist   
    os.makedirs(output_dir, exist_ok=True)
    
    with open('inference/t2vSafetyBench_prompt.txt', 'r') as file:
        prompt = file.read()
    
    extracted_frames = extract_frames(video_path, output_dir)
    frames=[]
    for filename in os.listdir(output_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_dir, filename)
            image = Image.open(img_path)
            frames.append(image)
    print("Extracted frames: {}".format(len(frames)))
    frame_count = -1
    frame_record = {}
    for frame in frames:
        frame_count += 1
        old_inputs = processor.process(
        images=[frame],
        text=prompt
        )
        
        # move inputs to the correct device and make a batch of size 1
        #inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Inference
        inputs = {}
        #import ipdb; ipdb.set_trace()
        for key in old_inputs.keys():
            if 'idx' in key:
                inputs[key] = old_inputs[key].unsqueeze(0).to(model.device)
            elif 'ids' in key:
                inputs[key] = old_inputs[key].unsqueeze(0).to(model.device)
            else:
                inputs[key] = old_inputs[key].unsqueeze(0).to(model.device, model.dtype)
            
        
        #import ipdb; ipdb.set_trace()
        


        attempts = 0
        while attempts < 5:
            #with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                #model.to(dtype=torch.bfloat16)
                #inputs["images"] = inputs["images"].to(torch.bfloat16)
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )
            generated_tokens = output[0,inputs['input_ids'].size(1):]
            gpt_response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            attempts += 1
            if ("ANS: Yes" in gpt_response or "ANS: No" in gpt_response) and "Yes: " in gpt_response and "No: " in gpt_response and gpt_response.count('%') == 2:
                break
            else:
                print("Attempt {} : MOLMO response: {}. Retrying...".format(attempts, gpt_response))
        print("Testing {}: Prompt {}: Attempts: {} for MOLMO response: {}".format(prompt_type, prompt_number, attempts, gpt_response))
        
        record = {
                "Testing": prompt_type, 
                "Prompt": prompt_number, 
                "Attempts": attempts,
                "MOLMO response": gpt_response, 
                "Orig_prompt": data["original_prompt"], 
                "GLM-4 refined Prompt": data["prompt"], 
                "categories": data["categories"],
                "inappropriate_percentage": data["inappropriate_percentage"],
                "nudity_percentage": data["nudity_percentage"],
                "prompt_toxicity": data["prompt_toxicity"],
                "Video_path": data["output_path"]
                }
        
        frame_record["frame_{}".format(frame_count)] = record
    # Save the record to a JSON file
    output_json_path = os.path.join(output_dir, "record.json")
    with open(output_json_path, 'w') as f:
        json.dump(frame_record, f, indent=4)

