import random
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import einops
from tqdm import tqdm
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

for interested_path in glob.glob("/ceph/sagnihot/projects/CogVideo/outputs/testing_t2vSafetyBench/using_MOLMO_7B/**/**/"):
#for interested_path in glob.glob("/ceph/sagnihot/projects/CogVideo/outputs/testing_t2vSafetyBench/using_MOLMO_7B/testing_nudity/**/"):
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
    working_dir = "./outputs/using_new_distance_molmo/testing_{}/prompt_{}/".format(prompt_type, prompt_number)
    working_dir = os.path.join(working_dir, "coverage_{}_components_{}".format(coverage, args.components))
    if os.path.exists(os.path.join(working_dir, 'plots', 'iterating_layers_for_difference_between_harmful_and_harmless.png')):
        print("Plots already exist: ", os.path.join(working_dir, 'plots', 'iterating_layers_for_difference_between_harmful_and_harmless.png'))
        continue
    else:
        os.makedirs(working_dir, exist_ok=True)
        local_repo_dir_plots = os.path.join(working_dir, 'plots')
        os.makedirs(local_repo_dir_plots, exist_ok=True)


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

        harmless_paths = glob.glob("/ceph/sagnihot/datasets/Graphically Safe Images/*")
        harmful_paths = glob.glob("/ceph/sagnihot/datasets/Graphically Violent Images/*")

        interested_paths = glob.glob(interested_path + "/*.jpg")

        harmful_instructions, harmless_instructions, interested_instructions = [], [], []
        for path in harmful_paths:
            if '.jpg' in path.lower() or '.png' in path.lower() or '.jpeg' in path.lower():
                harmful_instructions.append(Image.open(path))
        for path in harmless_paths:
            if '.jpg' in path.lower() or '.png' in path.lower() or '.jpeg' in path.lower():
                harmless_instructions.append(Image.open(path))
        for path in interested_paths:
            if '.jpg' in path.lower() or '.png' in path.lower() or '.jpeg' in path.lower():
                interested_instructions.append(Image.open(path))

        harmful = None
        harmless = None
        gc.collect()

        # Progress


        # Generate target layer hidden state files for harmful and harmless features
        def save_target_hidden_states(image, index, feature):
            bar.update(n=1)
            # We still select the target layers, then only keep the hidden state of the last token (-1 part)
            old_inputs = processor.process(
                images=image,
                text=prompt
                )
            inputs = {}
            #import ipdb; ipdb.set_trace()
            for key in old_inputs.keys():
                if 'idx' in key:
                    inputs[key] = old_inputs[key].unsqueeze(0).to(model.device)
                elif 'ids' in key:
                    inputs[key] = old_inputs[key].unsqueeze(0).to(model.device)
                else:
                    inputs[key] = old_inputs[key].unsqueeze(0).to(model.device, model.dtype)
                
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(stop_strings="<|endoftext|>", max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True),
                tokenizer=processor.tokenizer
            )
            
            hidden = torch.stack([layer[:, -1, :] for layer in output.hidden_states[0]], dim=0)
            # Squeeze away token dimension, remove token_embedding layer output? 
            hidden = hidden.squeeze(1)[1:, :]
            # Save each hidden state to disk to keep memory usage at a minimum
            dir_path = working_dir + "/" + feature + "_states"
            file_path = dir_path + "/" + str(index) + ".pt"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(hidden, file_path)

        # Save harmful states
        #"""
        max_its = len(harmless_paths) + len(harmful_paths) + len(interested_paths)
        bar = tqdm(total=max_its)
        for index, instruction in enumerate(harmful_instructions):
            save_target_hidden_states(instruction, index, "harmful")

        # Save harmless states
        for index, instruction in enumerate(harmless_instructions):
            save_target_hidden_states(instruction, index, "harmless")

        for index, instruction in enumerate(interested_instructions):
            save_target_hidden_states(instruction, index, "interested")
        # End progress bar
        bar.close()
        #"""
        # Clean-up
        model = None
        harmful_instructions = None
        harmless_instructions = None
        interested_instructions = None
        gc.collect()
        torch.cuda.empty_cache()

        #instructions = 512 #32
        n_components = args.components
        #n_layers = 40
        n_layers = 28

        gc.collect()
        torch.cuda.empty_cache()

        # Load tensors
        harmful_tensors = [torch.load(path, weights_only=True) for path in glob.glob(f"{working_dir}/harmful_states/*.pt")]
        harmless_tensors = [torch.load(path, weights_only=True) for path in glob.glob(f"{working_dir}/harmless_states/*.pt")]
        interested_tensors = [torch.load(path, weights_only=True) for path in glob.glob(f"{working_dir}/interested_states/*.pt")]

        # Create data
        harmful_data = torch.stack(harmful_tensors).to(torch.float16).cpu()
        harmless_data = torch.stack(harmless_tensors).to(torch.float16).cpu()
        interested_data = torch.stack(interested_tensors).to(torch.float16).cpu()

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
                if len(filtered_indices) == 0:
                    # Find the closest values to the z_score_threshold
                    closest_index = torch.abs(torch.abs(z_scores) - z_score_threshold).argmin()
                    filtered = tensor[closest_index].unsqueeze(0)
                else:
                    filtered = torch.index_select(tensor, 0, filtered_indices)
                
                try:
                    return (filtered.min(), filtered.max())
                except:
                    
                    #import ipdb; ipdb.set_trace()
                    return (tensor.min(), tensor.max())
            harmful_min, harmful_max = majority_bounds(harmful_pca[:, pca_index], 0)
            harmless_min, harmless_max = majority_bounds(harmless_pca[:, pca_index], 0)
            interested_min, interested_max = majority_bounds(interested_pca[:, pca_index], 0)
            
            # Plot
            row = int(i / 10) * 2
            col = i % 10
            y_height = harmful_pca.shape[0]
            y_range = range(y_height)
            ax[row, col].add_patch(plt.Rectangle((harmful_min, 0), harmful_max - harmful_min, y_height, color='red', alpha=0.5))
            ax[row, col].add_patch(plt.Rectangle((harmless_min, 0), harmless_max - harmless_min, y_height, color='blue', alpha=0.5))
            ax[row, col].add_patch(plt.Rectangle((interested_min, 0), interested_max - interested_min, y_height, color='yellow', alpha=0.5))
            if harmless_min > harmful_max:
                ax[row, col].add_patch(plt.Rectangle((harmful_max, 0), harmless_min - harmful_max, y_height, color=(0, 1, 0), alpha=1.0))
                gaps.append(harmless_min - harmful_max)
                
            elif harmful_min > harmless_max:
                ax[row, col].add_patch(plt.Rectangle((harmless_max, 0), harmful_min - harmless_max, y_height, color=(0, 1, 0), alpha=1.0))
                gaps.append(harmful_min - harmless_max)
                
            else:
                gaps.append(0)
                
            if interested_min > harmful_max:
                gaps_interested_harmful.append(interested_min - harmful_max)
            elif harmful_min > interested_max:
                gaps_interested_harmful.append(harmful_min - interested_max)
            else:
                gaps_interested_harmful.append(0)
            
            if interested_min > harmless_max:
                gaps_interested_harmless.append(interested_min - harmless_max)
            elif harmless_min > interested_max:
                gaps_interested_harmless.append(harmless_min - interested_max)
            else:
                gaps_interested_harmless.append(0)
                
            ax[row, col].scatter(harmful_pca[:, pca_index], y_range, color='red', s=8, label='Harmful')
            ax[row, col].scatter(harmless_pca[:, pca_index], range(harmless_pca.shape[0]), color='blue', s=8, label='Harmless')
            ax[row, col].scatter(interested_pca[:, pca_index], range(interested_pca.shape[0]), color='yellow', s=8, label='Interested')
            
            # Components Plot
            comp_row = row + 1
            x_range = range(pca.components_.shape[1])
            delta_components = None
            if i == 0:
                delta_components = pca_components[i][pca_index]
            else:
                delta_components = pca_components[i][pca_index]
                #delta_components = pca_components[i][pca_index] - pca_components[i-1][pca_index]
            #components_2 = pca_components_2[i][pca_index]
            ax[comp_row, col].plot(x_range, delta_components, color="red", alpha=0.5)
            #ax[comp_row, col].plot(x_range, components_2, color="blue", alpha=0.5)
            ax[comp_row, col].set_title(f"{delta_components.abs().argmax()}")
            ax[comp_row, col].set_ylim([-1, 1])
            
        # Remove un-used plot cells
        for i in range(n_layers, nrows * 10):
            row = int(i / 10) * 2
            col = i % 10
            comp_row = row + 1
            ax[row, col].set_title("")
            ax[row, col].axis("off")
            ax[comp_row, col].set_title("")
            ax[comp_row, col].axis("off")
            
        # Iterate through our layers until we detect separation between harmful and harmless
        layer_index = -1
        records = []
        for i in range(n_layers):
            row = int(i / 10) * 2
            col = i % 10
            if gaps[i] > 0 and layer_index < 0:
                ax[row, col].set_facecolor((0, 1, 0))
                layer_index = i
                ax[row, col].set_title(f"Layer {i} (target)")
                

                gaps_interested_harmful[i] = gaps_interested_harmful[i] / gaps[i]
                gaps_interested_harmless[i] = gaps_interested_harmless[i] / gaps[i]
                
                if gaps_interested_harmful[i] > 1 or  gaps_interested_harmless[i] > 1:
                    if gaps_interested_harmful[i] > gaps_interested_harmless[i]:
                        gaps_interested_harmful[i] = 1
                        gaps_interested_harmless[i] = 0
                    else:
                        gaps_interested_harmless[i] = 1
                        gaps_interested_harmful[i] = 0
                
                records.append({"Layer": i, "Harmful": gaps_interested_harmful[i], "Harmless": gaps_interested_harmless[i], "Gap": gaps[i]})
                
                print(f"Layer {i} (target) - Harmful: {gaps_interested_harmful[i]*100}% Harmless: {gaps_interested_harmless[i]*100}%")
            else:
                ax[row, col].set_facecolor((0, 0, 0))
                ax[row, col].set_title(f"Layer {i}")
            
            
        plt.tight_layout()
        plt.savefig(os.path.join(local_repo_dir_plots, "iterating_layers_for_difference_between_harmful_and_harmless.png"))
        plt.show()

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(local_repo_dir_plots, "layer_index_{}_gaps.csv".format(layer_index)), index=False)
        # Convert PCA components to PyTorch tensor
        pca_components = torch.stack(pca_components, dim=1)

        pca_components_mean = pca_components[pca_index][24].abs() #.abs()[24] #.mean(dim=0)
        plt.figure(figsize=(5, 4))
        plt.plot(range(pca_components_mean.shape[0]), pca_components_mean / pca_components_mean.norm(), color="red", alpha=0.5)
        #plt.show()

        # Instructions mean
        harmful_mean = harmful_data.mean(dim=0)
        harmless_mean = harmless_data.mean(dim=0)
        mean_diff = harmless_mean - harmful_mean #- harmless_mean

        mean_diff_mean = mean_diff[24].abs() #.mean(dim=0)
        #plt.figure(figsize=(5, 4))
        plt.plot(range(mean_diff_mean.shape[0]), mean_diff_mean / mean_diff_mean.norm(), color="blue", alpha=0.5)
        plt.savefig(os.path.join(local_repo_dir_plots, "principle_components.png"))
        plt.show()

        # Calculate cosine similarity using PyTorch
        cosine_similarities = torch.cosine_similarity(mean_diff.unsqueeze(0), pca_components, dim=-1)

        # Visualize cosine similarities
        plt.figure(figsize=(12, 4))
        plt.imshow(cosine_similarities, cmap='RdBu', interpolation='nearest', vmin=-1.0, vmax=1.0)
        cbar = plt.colorbar()
        cbar.set_ticks([-0.5, 0.0, 0.5])
        plt.xlabel('Layer')
        plt.ylabel('Component')
        plt.title('Cosine Similarity (Mean diff and PCs)')
        plt.savefig(os.path.join(local_repo_dir_plots, "cosine_similarity_mean_diff_and_principle_components.png"))
        plt.show()

        # DEBUG
        #layer_index = 20

        # Ideal layer index
        if layer_index == -1:
            layer_index = n_layers // 2
        print(f"\n\n\t\t\tUsing layer index: {layer_index}")

        
        # Clean-up
        cosine_similarities = None
        pca_components = None
        differences = None
        variances = None
        scores = None
        harmful_data = None
        harmless_data = None
        harmful_mean = None
        harmless_mean = None
        mean_diff = None
        gc.collect()