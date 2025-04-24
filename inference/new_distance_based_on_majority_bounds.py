import random
import os
import gc
import torch
#from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#from qwen_vl_utils import process_vision_info
#import spaces
import einops
from tqdm import tqdm
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

import argparse
from PIL import Image

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--coverage', type=float, default=0.5)
parser.add_argument('--instructions', type=int, default=5229)
parser.add_argument('--components', type=int, default=10)
parser.add_argument('--interested_path', type=str, default="/ceph/sagnihot/projects/CogVideo/outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_inappropriate/prompt_2788/")
parser.add_argument('--prompt', type=str, default="Is this image safe for work? Answer with yes or no.")

args = parser.parse_args()

def get_free_gpu_memory_gb():
    """
    Calculates the free GPU memory in GB for the current device.
    """
    if not torch.cuda.is_available():
        return "CUDA is not available."

    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = total_memory - allocated_memory
    return free_memory / (1024**3)

#for interested_path in glob.glob("outputs/testing_t2vSafetyBench/using_MOLMO_7B/**/**/"):
for interested_path in ["outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_nudity/prompt_3171/", "outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_inappropriate/prompt_2788"]:
    prompt_number = interested_path.split("prompt_")[-1].split("/")[0]
    prompt_type = interested_path.split("testing_")[-1].split("/")[0]
    # Clear memory of past model usage
    model = None
    tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

    torch.inference_mode()

    prompt = args.prompt

    coverage = args.coverage
    local_repo_dir = "./working/glm-4-9b-chat"
    working_dir = "./outputs/using_new_distance_QWEN_2.5/testing_{}/prompt_{}/".format(prompt_type, prompt_number)
    working_dir = os.path.join(working_dir, "coverage_{}_components_{}".format(coverage, args.components))
    #if os.path.exists(os.path.join(working_dir, 'plots', 'cosine_similarity_interested_harmless.png')):
    if False:
        print("Plots already exist: ", os.path.join(working_dir, 'plots', 'cosine_similarity_interested_harmless.png'))
        continue
    else:
        os.makedirs(working_dir, exist_ok=True)
        local_repo_dir_plots = os.path.join(working_dir, 'plots')
        os.makedirs(local_repo_dir_plots, exist_ok=True)
        """
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

        harmful = None
        harmless = None
        gc.collect()
        
        model_info = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_info , torch_dtype="bfloat16", device_map="cuda"
                    ).to(torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_info, trust_remote_code=True)

        # Progress


        # Generate target layer hidden state files for harmful and harmless features
        
        def save_target_hidden_states(instruction, index, feature):
            
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

            output = model.generate(**inputs, max_new_tokens=1, 
                                           return_dict_in_generate=True, 
                                           output_hidden_states=True, 
                                           use_cache=False)
            hidden_states = tuple(hs.cpu() for hs in output.hidden_states[0])
            hidden = torch.stack([layer[:, -1, :] for layer in hidden_states], dim=0)
            del output.hidden_states
            del output
            gc.collect()
            torch.cuda.empty_cache()
            hidden = hidden.squeeze(1)[1:, :]
            
            
            dir_path = working_dir + "/" + feature + "_states"
            file_path = dir_path + "/" + str(index) + ".pt"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(hidden, file_path)
            #del hidden
            #hidden = None
            gc.collect()
            torch.cuda.empty_cache()

        # Save harmful states
        
        max_its = len(harmless_paths) + len(harmful_paths) + len(interested_paths)
        bar = tqdm(total=max_its)
        for index, instruction in enumerate(harmful_instructions):
            try:
                save_target_hidden_states(instruction, index, "harmful")
                bar.set_postfix_str(" Harmful Done: {}".format(os.path.basename(instruction)))
                bar.update(1)
            except Exception as e:
                bar.set_postfix_str("Harmful SKIPPED: {}".format(os.path.basename(instruction)))
                bar.update(1)

        # Save harmless states
        for index, instruction in enumerate(harmless_instructions):
            try:
                save_target_hidden_states(instruction, index, "harmless")
                bar.set_postfix_str("Harless Done: {}".format(os.path.basename(instruction)))
                bar.update(1)
            except Exception as e:
                bar.set_postfix_str("Harmless SKIPPED: {}".format(os.path.basename(instruction)))
                bar.update(1)

        for index, instruction in enumerate(interested_instructions):
            try:
                save_target_hidden_states(instruction, index, "interested")
                bar.set_postfix_str("Interested Done: {}".format(os.path.basename(instruction)))
                bar.update(1)
            except Exception as e:
                bar.set_postfix_str("Interested SKIPPED: {}".format(os.path.basename(instruction)))
                bar.update(1)
        # End progress bar
        bar.close()
        """
        # Clean-up
        model = None
        processor = None
        harmful_instructions = None
        harmless_instructions = None
        interested_instructions = None
        gc.collect()
        torch.cuda.empty_cache()

        #instructions = 512 #32
        n_components = args.components
        #n_layers = 40
        

        gc.collect()
        torch.cuda.empty_cache()

        # Load tensors
        tensor_working_dir = working_dir.replace("coverage_{}_components_{}".format(coverage, args.components), "coverage_{}_components_{}".format(0.5, 10))
        harmful_tensors = [torch.load(path, weights_only=True) for path in glob.glob(f"{tensor_working_dir}/harmful_states/*.pt")]
        harmless_tensors = [torch.load(path, weights_only=True) for path in glob.glob(f"{tensor_working_dir}/harmless_states/*.pt")]
        interested_tensors = [torch.load(path, weights_only=True) for path in glob.glob(f"{tensor_working_dir}/interested_states/*.pt")]

        # Create data
        harmful_data = torch.stack(harmful_tensors).to(torch.float16).cpu()
        harmless_data = torch.stack(harmless_tensors).to(torch.float16).cpu()
        interested_data = torch.stack(interested_tensors).to(torch.float16).cpu()
        
        n_layers = harmful_data.shape[1]

        harmful_tensors = None
        harmless_tensors = None
        interested_tensors = None
        gc.collect()
        torch.cuda.empty_cache()

        pca_components = []
        gaps = []
        gaps_interested_harmless, gaps_interested_harmful = [], []

        # We can create a majority region of our PCAs by removing the outliers via z-score thresholding
        # Once the two regions (harmful and harmless PCA 1st component) are separated we know refusal has been introduced
        # The amount of separation that we deem to be "enough" can be controlled by our coverage hyper-parameter
        # Calculate our z-score threshold based on coverage
        #coverage = 0.75
        #coverage = 0.9

        # Inverse CDF on normal distribution with probability equal to our coverage, both tail ends will be trimmed so icdf is used accordingly
        z_score_threshold = torch.distributions.normal.Normal(loc=0, scale=1).icdf(torch.tensor([coverage + (1 - coverage) / 2])).item()
        #print(f"Using z-score threshold: {z_score_threshold}")

        # Plot
        pca_index = 0 #0 #1
        plots_per_layer = 2
        nrows = math.ceil(n_layers / 10)
        ncols = 10
        fig, ax = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(5 * 10 // 2, 4 * nrows * 2 // 2))
        harmful_sort = []
        harmless_sort = []
        interested_sort = []
        pca = PCA(n_components=n_components)
        for i in range(n_layers):
            # PCA
            #pca = PCA(n_components=n_components)
            #import ipdb; ipdb.set_trace()
            harmful_pca = torch.tensor(pca.fit_transform(harmful_data[:, i, :]))
            harmless_pca = torch.tensor(pca.transform(harmless_data[:, i, :]))
            interested_pca = torch.tensor(pca.transform(interested_data[:, i, :]))
            pca_components.append(torch.tensor(pca.components_))
            
            # Sort sample instructions for cleaner starting visual
            if i == 0:
                harmful_sort = torch.argsort(harmful_pca[:, 0], descending=False)
                harmless_sort = torch.argsort(harmless_pca[:, 0], descending=False)
                interested_sort = torch.argsort(interested_pca[:, 0], descending=False)
            harmful_pca = harmful_pca[harmful_sort]
            harmless_pca = harmless_pca[harmless_sort]
            interested_pca = interested_pca[interested_sort]
            
            # Find max and min excluding outliers using Z-score
            # Coverage is a normalized percentage of included elements based on a normal distribution, 99.73% (0.9973) would be a z_score of 3
            def majority_bounds(tensor, pca_index, z_score_threshold=z_score_threshold):
                z_scores = (tensor - tensor.mean()) / tensor.std()
                filtered_indices = torch.where(torch.abs(z_scores) < z_score_threshold)[0]
                filtered = torch.index_select(tensor, 0, filtered_indices)
                
                try:
                    return (filtered.min(), filtered.max())
                except:
                    import ipdb; ipdb.set_trace()
                    return (tensor.min(), tensor.max())
            harmful_min, harmful_max = majority_bounds(harmful_pca[:, pca_index], 0)
            harmless_min, harmless_max = majority_bounds(harmless_pca[:, pca_index], 0)
            interested_min, interested_max = majority_bounds(interested_pca[:, pca_index], 0)
            
            row = int(i / 10) * 2
            col = i % 10
            y_height = harmful_pca.shape[0]
            y_range = range(y_height)
            ax[row, col].add_patch(plt.Rectangle((harmful_min, 0), harmful_max - harmful_min, y_height, color='red', alpha=0.5))
            ax[row, col].add_patch(plt.Rectangle((harmless_min, 0), harmless_max - harmless_min, y_height, color='blue', alpha=0.5))
            ax[row, col].add_patch(plt.Rectangle((interested_min, 0), interested_max - interested_min, y_height, color='yellow', alpha=0.5))
            
            if harmful_min>harmless_max:
                ax[row, col].set_title("Layer {} Harmful Greater: {}".format(i, harmful_min-harmless_max))
            if harmless_min>harmful_max:
                ax[row, col].set_title("Layer {} Harmless Greater: {}".format(i, harmless_min-harmful_max))
                
        plt.tight_layout()
        plt.savefig(os.path.join(local_repo_dir_plots, "PATCHES_iterating_layers_for_difference_between_harmful_and_harmless.png"))
        plt.show()