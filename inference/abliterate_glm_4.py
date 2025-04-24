from huggingface_hub import snapshot_download
snapshot_download(repo_id="THUDM/glm-4-9b-chat", local_dir="./working/glm-4-9b-chat")

import jaxtyping
import random
import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import einops
from tqdm import tqdm

import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('--coverage', type=float, default=0.5)
parser.add_argument('--instructions', type=int, default=5229)
parser.add_argument('--components', type=int, default=10)

args = parser.parse_args()

# Clear memory of past model usage
model = None
tokenizer = None
gc.collect()
torch.cuda.empty_cache()

torch.inference_mode()

coverage = args.coverage
local_repo_dir = "./working/glm-4-9b-chat"
working_dir = "./working"
working_dir = os.path.join(working_dir, "coverage_{}_components_{}".format(coverage, args.components))
os.makedirs(working_dir, exist_ok=True)
local_repo_dir_plots = os.path.join(working_dir, "coverage_{}".format(coverage), 'plots')
os.makedirs(local_repo_dir_plots, exist_ok=True)


# Load model with only num_layers we actually need for this step
model = AutoModelForCausalLM.from_pretrained(local_repo_dir, local_files_only=True, trust_remote_code=True, 
                                             torch_dtype=torch.float16, 
                                             #num_layers=layer_idx+1,
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True, 
                                                                                    bnb_4bit_compute_dtype=torch.float16))
tokenizer = AutoTokenizer.from_pretrained(local_repo_dir, local_files_only=True, trust_remote_code=True)

# Settings
# I have used 128 and 256 with success but may as well use the max for a better estimation
instructions = args.instructions
#layer_idx = int(len(model.model.layers) * 0.5) #6)

print("Instruction count: " + str(instructions))

with open("./working/remove-refusals-with-transformers/harmful_new.txt", "r") as f:
    harmful = f.readlines()

with open("./working/remove-refusals-with-transformers/harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, instructions)
harmless_instructions = random.sample(harmless, instructions)

harmful = None
harmless = None
gc.collect()

# Progress
max_its = instructions * 2
bar = tqdm(total=max_its)

# Generate target layer hidden state files for harmful and harmless features
def save_target_hidden_states(prompt, index, feature):
    bar.update(n=1)
    toks = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt")
    # Generates using each example, cache is disables so it doesn't keep previous examples in it's context, obviously we need to output the full states
    # It would be ideal if we could have it output the states for only the layer we want
    output = model.generate(toks.to(model.device), use_cache=False, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)
    # We still select the target layers, then only keep the hidden state of the last token (-1 part)
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
for index, instruction in enumerate(harmful_instructions):
    save_target_hidden_states(instruction, index, "harmful")

# Save harmless states
for index, instruction in enumerate(harmless_instructions):
    save_target_hidden_states(instruction, index, "harmless")

# End progress bar
bar.close()

# Clean-up
model = None
harmful_instructions = None
harmless_instructions = None
gc.collect()
torch.cuda.empty_cache()

#instructions = 512 #32
n_components = args.components
n_layers = 40

gc.collect()
torch.cuda.empty_cache()

# Load tensors
harmful_tensors = [torch.load(f"{working_dir}/harmful_states/{i}.pt", weights_only=True) for i in range(instructions)]
harmless_tensors = [torch.load(f"{working_dir}/harmless_states/{i}.pt", weights_only=True) for i in range(instructions)]

# Create data
harmful_data = torch.stack(harmful_tensors).cpu()
harmless_data = torch.stack(harmless_tensors).cpu()

harmful_tensors = None
harmless_tensors = None
gc.collect()
torch.cuda.empty_cache()

pca_components = []
gaps = []

# We can create a majority region of our PCAs by removing the outliers via z-score thresholding
# Once the two regions (harmful and harmless PCA 1st component) are separated we know refusal has been introduced
# The amount of separation that we deem to be "enough" can be controlled by our coverage hyper-parameter
# Calculate our z-score threshold based on coverage
#coverage = 0.75
#coverage = 0.9

# Inverse CDF on normal distribution with probability equal to our coverage, both tail ends will be trimmed so icdf is used accordingly
z_score_threshold = torch.distributions.normal.Normal(loc=0, scale=1).icdf(torch.tensor([coverage + (1 - coverage) / 2])).item()
print(f"Using z-score threshold: {z_score_threshold}")

# Plot
pca_index = 0 #0 #1
plots_per_layer = 2
nrows = math.ceil(n_layers / 10)
ncols = 10
fig, ax = plt.subplots(nrows=nrows * 2, ncols=ncols, figsize=(5 * 10 // 2, 4 * nrows * 2 // 2))
harmful_sort = []
harmless_sort = []
pca = PCA(n_components=n_components)
for i in range(n_layers):
    # PCA
    #pca = PCA(n_components=n_components)
    harmful_pca = torch.tensor(pca.fit_transform(harmful_data[:, i, :]))
    harmless_pca = torch.tensor(pca.transform(harmless_data[:, i, :]))
    pca_components.append(torch.tensor(pca.components_))
    
    # Sort sample instructions for cleaner starting visual
    if i == 0:
        harmful_sort = torch.argsort(harmful_pca[:, 0], descending=False)
        harmless_sort = torch.argsort(harmless_pca[:, 0], descending=False)
    harmful_pca = harmful_pca[harmful_sort]
    harmless_pca = harmless_pca[harmless_sort]
    
    # Find max and min excluding outliers using Z-score
    # Coverage is a normalized percentage of included elements based on a normal distribution, 99.73% (0.9973) would be a z_score of 3
    def majority_bounds(tensor, pca_index, z_score_threshold=z_score_threshold):
        z_scores = (tensor - tensor.mean()) / tensor.std()
        filtered_indices = torch.where(torch.abs(z_scores) < z_score_threshold)[0]
        filtered = torch.index_select(tensor, 0, filtered_indices)
        return (filtered.min(), filtered.max())
    harmful_min, harmful_max = majority_bounds(harmful_pca[:, pca_index], 0)
    harmless_min, harmless_max = majority_bounds(harmless_pca[:, pca_index], 0)
    
    # Plot
    row = int(i / 10) * 2
    col = i % 10
    y_height = harmful_pca.shape[0]
    y_range = range(y_height)
    ax[row, col].add_patch(plt.Rectangle((harmful_min, 0), harmful_max - harmful_min, y_height, color='red', alpha=0.5))
    ax[row, col].add_patch(plt.Rectangle((harmless_min, 0), harmless_max - harmless_min, y_height, color='blue', alpha=0.5))
    if harmless_min > harmful_max:
        ax[row, col].add_patch(plt.Rectangle((harmful_max, 0), harmless_min - harmful_max, y_height, color=(0, 1, 0), alpha=1.0))
        gaps.append(harmless_min - harmful_max)
    elif harmful_min > harmless_max:
        ax[row, col].add_patch(plt.Rectangle((harmless_max, 0), harmful_min - harmless_max, y_height, color=(0, 1, 0), alpha=1.0))
        gaps.append(harmful_min - harmless_max)
    else:
        gaps.append(0)
    ax[row, col].scatter(harmful_pca[:, pca_index], y_range, color='red', s=8, label='Harmful')
    ax[row, col].scatter(harmless_pca[:, pca_index], y_range, color='blue', s=8, label='Harmless')
    
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
for i in range(n_layers):
    row = int(i / 10) * 2
    col = i % 10
    if gaps[i] > 0 and layer_index < 0:
        ax[row, col].set_facecolor((0, 1, 0))
        layer_index = i
        ax[row, col].set_title(f"Layer {i} (target)")
    else:
        ax[row, col].set_facecolor((0, 0, 0))
        ax[row, col].set_title(f"Layer {i}")
    
    
plt.tight_layout()
plt.savefig(os.path.join(local_repo_dir_plots, "iterating_layers_for_difference_between_harmful_and_harmless.png"))
plt.show()

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

# Save ideal layer mean_diff as refusal direction
mean_diff = -(mean_diff[layer_index])
refusal_direction = mean_diff / mean_diff.norm()

# Test targeting features
#manual_direction = torch.zeros(4096, dtype=torch.float16)
#manual_direction[3584] = 1.0
#refusal_direction = manual_direction

count = 0
for i in range(refusal_direction.shape[0]):
    if refusal_direction[i].abs() < 0.000:
        refusal_direction[i] = 0
        count += 1
print(f"Removed {count} of Refusal direction {refusal_direction.shape[0]} embed dims.")

plt.figure(figsize=(5, 4))
plt.plot(range(refusal_direction.shape[0]), refusal_direction, color="red", alpha=0.5)
plt.savefig(os.path.join(local_repo_dir_plots, "refusal_direction_after_removal.png"))
plt.show()

print(refusal_direction)
if not os.path.exists(local_repo_dir):
    os.makedirs(local_repo_dir)
torch.save(refusal_direction, working_dir + "/" + "refusal_direction.pt")

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