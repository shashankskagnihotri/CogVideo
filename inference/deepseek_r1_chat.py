import torch
import gradio as gr
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
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

model_id = "deepseek-ai/deepseek-vl-7b-chat"  # Check for latest version
#model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()



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

    output_dir = "outputs/testing_t2vSafetyBench/using_deepseek_r1/testing_{}/prompt_{}".format(prompt_type, prompt_number)
    # Create the output directory if it doesn't exist   
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames from the video
    extracted_frames = extract_frames(video_path, output_dir)



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
        
        
    conversation = [
        {
            "role": "User",
            "content": prompt,
            "images": extracted_frames
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]
        
    # Process inputs
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=frames,
        force_batchify=True
    ).to(vl_gpt.device)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Call the videochat function with the prompt and an example image (replace 'example_image.jpg' with your image path)
    attempts = 0
    while attempts < 5:
        # Generate output
        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        # Decode output
        gpt_response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)                                                                            
        attempts += 1
        if ("ANS: Yes" in gpt_response or "ANS: No" in gpt_response) and "Yes: " in gpt_response and "No: " in gpt_response and gpt_response.count('%') == 2:
            break
        else:
            print("Attempt {} : DeepSeek-R1 response: {}. Retrying...".format(attempts, gpt_response))
    print("Testing {}: Prompt {}: Attempts: {} for DeepSeek-R1 response: {}".format(prompt_type, prompt_number, attempts, gpt_response))
    
    record = {
            "Testing": prompt_type, 
              "Prompt": prompt_number, 
              "Attempts": attempts,
              "DeepSeek-R1 response": gpt_response, 
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

