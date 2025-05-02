import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
import glob
import ipdb
from tqdm import tqdm

import matplotlib.pyplot as plt

# Define paths
base_path = "/ceph/sagnihot/projects/CogVideo/outputs/hidden_states_latent"
plot_save_path = os.path.join(base_path, "plots")
os.makedirs(plot_save_path, exist_ok=True)

# Define configurations
configs = [
    #("testing_abliterated_glm_without_prompt_prefix", "using_glm"),
    ("testing_baseline_without_prompt_prefix", None)
]
negative_prompt_options = ["using_negative_prompt", "without_negative_prompt"]
safety_options = ["testing_safe", "testing_unsafe"]
#num_prompts = 10  # Assuming prompt_0 to prompt_9
num_steps = 50  # Number of inference steps

# Helper function to load embeddings
def load_embeddings(config, negative_prompt, safety, prompt, step):
    path = os.path.join(
        base_path,
        config[0],
        "testing_gpt4_nudity",
        negative_prompt,
        safety,
        config[1] if config[1] else "",
        f"prompt_{prompt}",
        f"output_{prompt}_hidden_states",
        f"latents_step_{step}.pt"
    )
    #path = glob.glob(path)[0] if len(glob.glob(path)) == 0 else ipdb.set_trace()
    if os.path.exists(path):
        return torch.load(path, map_location='cpu').to(torch.float16).cpu().numpy()
    else:
        print(f"File not found: {path}")
        return None

# Helper function to plot embeddings
def plot_embeddings(embeddings, labels, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    pca = PCA(n_components=2)
    pca_10 = PCA(n_components=10)
    #tsne = TSNE(n_components=2, perplexity=5, random_state=42)

    embeddings_pca = pca.fit_transform(embeddings)
    embeddings_pca_10 = pca_10.fit_transform(embeddings)
    #embeddings_tsne = tsne.fit_transform(embeddings)
    

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #for ax, data, method in zip(axes, [embeddings_pca, embeddings_tsne], ["PCA", "t-SNE"]):
    for ax, data, method in zip(axes, [embeddings_pca, embeddings_pca_10], ["PCA-2", "PCA-10"]):
        for label, color in [("safe", "blue"), ("unsafe", "red")]:
            idx = [i for i, lbl in enumerate(labels) if lbl == label]
            ax.scatter(data[idx, 0], data[idx, 1], c=color, label=label, alpha=0.6)
        if 'abliterated' in title:
            if "without_negative_prompt" in title:
                name = "abliterated GLM-4"
            else:
                name = "abliterated GLM-4 + -ve prompt"
        else:
            if "without_negative_prompt" in title:
                name = "baseline + -ve prompt"
            else:
                name = "baseline"
        ax.set_title(f"{method} - {name}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    #print(f"Plot saved at: {save_path}")
    plt.close()

# Main processing loop

for config in configs:
    for negative_prompt in negative_prompt_options:
        prompt_base_path = os.path.join(
                base_path,
                config[0],
                "testing_gpt4_nudity",
                negative_prompt
            )
        
        if 'abliterated' in config[0]:
            name = "abliterated GLM-4"
            prompt_folders = glob.glob(os.path.join(prompt_base_path, "*", "*", "prompt_*"))
        else:
            name = "baseline"
            prompt_folders = glob.glob(os.path.join(prompt_base_path, "*", "prompt_*"))
        for step in tqdm(range(num_steps), desc=f"Processing {name} - {negative_prompt}"):
            embeddings = []
            labels = []
            
            #import ipdb; ipdb.set_trace()
            
            if len(prompt_folders) == 0:
                print(f"No prompt folders found for {config[0]} with {negative_prompt} at step {step}.")
                import ipdb; ipdb.set_trace()
            for prompt_folder in prompt_folders:
                safety = "testing_safe" if "testing_safe" in prompt_folder else "testing_unsafe"
                prompt = os.path.basename(prompt_folder)
                embedding = load_embeddings(config, negative_prompt, safety, prompt.split('_')[-1], step)
                if embedding is not None:
                    embeddings.append(embedding.flatten())
                    labels.append("safe" if safety == "testing_safe" else "unsafe")

            if embeddings:
                embeddings = np.array(embeddings)
                title = f"{config[0]} - {negative_prompt} - Step {step}"
                save_path = os.path.join(plot_save_path, f"{config[0]}_{negative_prompt}", f"step_{step}.png")
                #import ipdb; ipdb.set_trace()
                if embeddings.shape[0] > 5:
                    plot_embeddings(embeddings, labels, title, save_path)
                

# Function to create a video from plots
def create_video_from_plots(root_folder):
    # Get all unique categories by parsing filenames
    plot_files = glob.glob(os.path.join(root_folder, '*', "*.png"))
    categories = set(os.path.basename(os.path.dirname(f)) for f in plot_files)

    for category in categories:
        images = []
        for step in range(50):  # Step count from 0 to 49
            plot_path = os.path.join(root_folder, f"{category}_step_{step}.png")
            if os.path.exists(plot_path):
                images.append(cv2.imread(plot_path))
            else:
                print(f"Plot not found: {plot_path}")

        if images:
            video_path = os.path.join(root_folder, f"{category}.mp4")
            height, width, _ = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))  # 2 FPS

            for img in images:
                video.write(img)
            video.release()
            print(f"Video saved at: {video_path}")
        else:
            print(f"No images found for category {category} to create a video.")


# Create video for the current combination
create_video_from_plots("/ceph/sagnihot/projects/CogVideo/outputs/hidden_states_latent/plots")
