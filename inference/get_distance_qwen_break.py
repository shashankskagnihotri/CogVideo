import torch
import gradio as gr
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import spaces
import cv2
import os
import argparse
from PIL import Image
import glob
import json
import subprocess

#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
REQUIRED PACKAGES:
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process video and generate captions.")
parser.add_argument("--video_path", type=str, required=False, default='outputs/testing_ablitearted_glm4_without_prompt_prefix/testing_nudity/using_glm/prompt_2110/output_2110.mp4', help="Path to the input video file.")
parser.add_argument('--coverage', type=float, default=0.5)
parser.add_argument('--instructions', type=int, default=5229)
parser.add_argument('--components', type=int, default=10)
parser.add_argument('--interested_path', type=str, default="/ceph/sagnihot/projects/CogVideo/outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_inappropriate/prompt_2788/")
parser.add_argument('--prompt', type=str, default="Is this image safe for work? Answer with yes or no.")
args = parser.parse_args()


prompt = args.prompt

coverage = args.coverage


#for interested_path in glob.glob("outputs/testing_t2vSafetyBench/using_MOLMO_7B/**/**/"):
for interested_path in ["outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_nudity/prompt_3171/", "outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_inappropriate/prompt_2788"]:
    prompt_number = interested_path.split("prompt_")[-1].split("/")[0]
    prompt_type = interested_path.split("testing_")[-1].split("/")[0]
    
    working_dir = "./outputs/using_new_distance_QWEN_2.5/testing_{}/prompt_{}/".format(prompt_type, prompt_number)
    working_dir = os.path.join(working_dir, "coverage_{}_components_{}".format(coverage, args.components))
    if os.path.exists(os.path.join(working_dir, 'plots', 'iterating_layers_for_difference_between_harmful_and_harmless.png')):
        print("Plots already exist: ", os.path.join(working_dir, 'plots', 'iterating_layers_for_difference_between_harmful_and_harmless.png'))
        continue
    else:
        os.makedirs(working_dir, exist_ok=True)
        local_repo_dir_plots = os.path.join(working_dir, 'plots')
        os.makedirs(local_repo_dir_plots, exist_ok=True)
        
        
        harmless_paths = glob.glob("/ceph/sagnihot/datasets/Graphically Safe Images/*")
        harmful_paths = glob.glob("/ceph/sagnihot/datasets/Graphically Violent Images/*")

        interested_paths = glob.glob(interested_path + "/*.jpg")

        harmful_instructions, harmless_instructions, interested_instructions = [], [], []
        for path in harmful_paths:
            if '.jpg' in path.lower() or '.png' in path.lower() or '.jpeg' in path.lower():
                harmful_instructions.append(path)
        for path in harmless_paths:
            if '.jpg' in path.lower() or '.png' in path.lower() or '.jpeg' in path.lower():
                harmless_instructions.append(path)
        for path in interested_paths:
            if '.jpg' in path.lower() or '.png' in path.lower() or '.jpeg' in path.lower():
                interested_instructions.append(path)
                
        for index, instruction in enumerate(harmful_instructions):
            feature = "harmful"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": instruction,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            dir_path = working_dir + "/" + feature + "_states"
            file_path = dir_path + "/" + str(index) + ".pt"
            subprocess.run(["python", "inference/get_qwen_hidden_states.py", 
                            "--instruction", instruction, "--feature", feature, 
                            "--working_dir", working_dir, "--prompt", prompt,
                            "--feature", feature, "--index", str(index)])
            print('Saved: {}'.format(file_path))
        
        for index, instruction in enumerate(harmless_instructions):
            feature = "harmless"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": instruction,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            dir_path = working_dir + "/" + feature + "_states"
            file_path = dir_path + "/" + str(index) + ".pt"
            subprocess.run(["python", "inference/get_qwen_hidden_states.py", 
                            "--instruction", instruction, "--feature", feature, 
                            "--working_dir", working_dir, "--prompt", prompt,
                            "--feature", feature, "--index", str(index)])
            print('Saved: {}'.format(file_path))
            
        for index, instruction in enumerate(interested_instructions):
            feature = "interested"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": instruction,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            dir_path = working_dir + "/" + feature + "_states"
            file_path = dir_path + "/" + str(index) + ".pt"
            subprocess.run(["python", "inference/get_qwen_hidden_states.py", 
                            "--instruction", instruction, "--feature", feature, 
                            "--working_dir", working_dir, "--prompt", prompt,
                            "--feature", feature, "--index", str(index)])
            print('Saved: {}'.format(file_path))
