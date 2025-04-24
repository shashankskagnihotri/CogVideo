import torch
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
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


#model_info = "Qwen/Qwen2.5-VL-32B-Instruct"
model_info = "Qwen/Qwen2.5-VL-7B-Instruct"

# Load pre-trained models for image captioning and language modeling
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_info , torch_dtype="bfloat16", device_map="cuda"
    )#.to(torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_info, trust_remote_code=True)


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


#video_paths = glob.glob('outputs/testing_ablitearted_glm4_without_prompt_prefix/**/using_glm/**/*.mp4')
video_paths = glob.glob('outputs/testing_basemodel_without_glm/testing_inappropriate/prompt_2752/output_2752.mp4')
for video_path in video_paths:
    prompt_number = video_path.split("prompt_")[-1].split("/")[0]
    prompt_type = video_path.split("testing_")[-1].split("/")[0]
    
    json_path = video_path.replace("output_{}.mp4".format(prompt_number), "record.json")
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    output_dir = "outputs/testing_t2vSafetyBench/using_QWEN_2.5_7B/testing_{}_generated_wtihout_glm/prompt_{}".format(prompt_type, prompt_number)
    # Create the output directory if it doesn't exist   
    os.makedirs(output_dir, exist_ok=True)
    
    with open('inference/t2vSafetyBench_prompt.txt', 'r') as file:
        prompt = file.read()
    
    # Extract frames from the video
    extracted_frames = extract_frames(video_path, output_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": extracted_frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    #image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=1.0,
        padding=True,
        return_tensors="pt",
        #**video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Inference
    


    attempts = 0
    while attempts < 5:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        gpt_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        attempts += 1
        if ("ANS: Yes" in gpt_response or "ANS: No" in gpt_response) and "Yes: " in gpt_response and "No: " in gpt_response and gpt_response.count('%') == 2:
            break
        else:
            print("Attempt {} : QWEN_2.5 response: {}. Retrying...".format(attempts, gpt_response))
    print("Testing {}: Prompt {}: Attempts: {} for QWEN_2.5 response: {}".format(prompt_type, prompt_number, attempts, gpt_response))
    
    record = {
            "Testing": prompt_type, 
              "Prompt": prompt_number, 
              "Attempts": attempts,
              "QWEN_2.5 response": gpt_response, 
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

