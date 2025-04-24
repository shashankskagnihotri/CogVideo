import torch
import gradio as gr
from transformers import AutoModel
from transformers import AutoProcessor
import spaces
import cv2
import os
import argparse
from PIL import Image
import glob
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process video and generate captions.")
parser.add_argument("--video_path", type=str, required=False, default='outputs/testing_ablitearted_glm4_without_prompt_prefix/testing_nudity/using_glm/prompt_2110/output_2110.mp4', help="Path to the input video file.")
args = parser.parse_args()

# Load pre-trained models for image captioning and language modeling
model3 = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True).to(torch.bfloat16)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)

# Define a function for image captioning
@spaces.GPU(queue=False)
def videochat(image3, prompt3):
    # Process input image and prompt
    inputs = processor(text=[prompt3], images=image3, return_tensors="pt")
    # Generate captions
    #import ipdb; ipdb.set_trace()
    inputs['images'] = inputs['images'].to(torch.bfloat16)
    with torch.inference_mode():
        output = model3.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )
        prompt_len = inputs["input_ids"].shape[1]
    # Decode and return the generated captions
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    if decoded_text.endswith("<|im_end|>"):
        decoded_text = decoded_text[:-10]
    return decoded_text
    

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
    #return extracted_frames


video_paths = glob.glob('outputs/testing_ablitearted_glm4_without_prompt_prefix/**/using_glm/**/*.mp4')
for video_path in video_paths:
    prompt_number = video_path.split("prompt_")[-1].split("/")[0]
    prompt_type = video_path.split("testing_")[-1].split("/")[0]
    
    json_path = video_path.replace("output_{}.mp4".format(prompt_number), "record.json")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_dir = "outputs/testing_t2vSafetyBench/using_opensource_gpt4o/testing_{}/prompt_{}".format(prompt_type, prompt_number)
    # Create the output directory if it doesn't exist   
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames from the video
    extract_frames(video_path, output_dir)



    frames=[]
    for filename in os.listdir(output_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(output_dir, filename)
            image = Image.open(img_path)
            frames.append(image)
    print("Extracted frames: {}".format(len(frames)))
    # Read the contents of the text file and save it as a string in the variable 'prompt'
    with open('inference/t2vSafetyBench_prompt.txt', 'r') as file:
        prompt = file.read()

    # Call the videochat function with the prompt and an example image (replace 'example_image.jpg' with your image path)
    attempts = 0
    while attempts<21:
        gpt_response = videochat(frames, prompt)
        attempts += 1
        if ("ANS: Yes" in gpt_response or "ANS: No" in gpt_response) and "Yes: " in gpt_response and "No: " in gpt_response and gpt_response.count('%') == 2:
            break
        else:
            print("Attempt {} : GPT-4o response: {}. Retrying...".format(attempts, gpt_response))
    print("Testing {}: Prompt {}: Attempts: {} for GPT-4o response: {}".format(prompt_type, prompt_number, attempts, gpt_response))
    
    record = {
            "Testing": prompt_type, 
              "Prompt": prompt_number, 
              "Attempts": attempts,
              "GPT-4o response": gpt_response, 
              "Orig_prompt": data["original_prompt"], 
              "GLM-4 refined Prompt": data["prompt"], 
              "categories": data["categories"],
              "inappropriate_percentage": data["inappropriate_percentage"],
              "nudity_percentage": data["nudity_percentage"],
              "prompt_toxicity": data["prompt_toxicity"],
              "Video_path": data["output_path"]
              }
    # Save the record to a JSON file
    output_json_path = os.path.join(output_dir, "record.json")
    with open(output_json_path, 'w') as f:
        json.dump(record, f, indent=4)

