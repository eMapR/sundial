import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations


BASE_PATH = os.getenv("BASE_PATH")
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)
CKPT_COMPARE_PATH = os.path.join(BASE_PATH, "ckpt_compare")


def filter_layers(state_dict, layer_pattern="encoder.blocks"):
    out = {}
    for k, v in state_dict.items():
        if layer_pattern in k and "num_batches_tracked" not in k and "running_mean" not in k and "running_var" not in k:
            if "prithvi" in k:
                out[k.replace("prithvi.", "")] = v
            else:
                out[k] = v
    return out
    
def compute_pairwise_distances(state_dicts):
    num_models = len(state_dicts)
    l2_distances = np.zeros((num_models, num_models))
    l1_distances = np.zeros((num_models, num_models))
    cosine_sims = np.zeros((num_models, num_models))
    cos = torch.nn.CosineSimilarity(dim=1)
    
    for (a, b), (i, j) in zip(combinations(state_dicts.keys(), 2), combinations(range(num_models), 2)):
        l1_dist = 0
        l2_dist = 0
        cos_sum = 0
        count = 0
        for (name1, param1), (name2, param2) in zip(state_dicts[a].items(), state_dicts[b].items()):
            diff = param1.view(-1).unsqueeze(0) - param2.view(-1).unsqueeze(0)
            l2_dist += torch.linalg.norm(diff, ord=2).item()
            l1_dist += torch.linalg.norm(diff, ord=1).item()
            cos_sum += cos(param1.view(-1).unsqueeze(0), param2.view(-1).unsqueeze(0)).item()
            count += 1
        
        l2_dist = l2_dist / count
        l1_dist = l1_dist / count
        cos_sum = cos_sum / count
        
        l2_distances[i, j] = l2_dist
        l2_distances[j, i] = l2_dist
        l1_distances[i, j] = l1_dist
        l1_distances[j, i] = l1_dist
        cosine_sims[i, j] = cos_sum
        cosine_sims[j, i] = cos_sum
    
    return l2_distances, l1_distances, cosine_sims

def plot_distance_matrix(distances, path, title, labels=None, rev=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(distances, cmap="coolwarm_r" if rev else "coolwarm")
     
    fig.colorbar(cax)

    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)  

    for i in range(len(distances)):
        for j in range(len(distances)):
            ax.text(j, i, f"{distances[i, j]:.3f}", ha='center', va='center', color='black')

    plt.title(title)
    plt.tight_layout(pad=2.0)  
    plt.savefig(path, dpi=300)

if __name__ == "__main__":
    random = torch.load("/ceoas/emapr/sundial/checkpoints/all224_glkn/random.pt", map_location=torch.device('cpu'), weights_only=True)
    frozen = torch.load("/ceoas/emapr/sundial/checkpoints/all224_glkn/epoch=0124_val_loss=0.354_fcn_dice_frozen.ckpt", map_location=torch.device('cpu'), weights_only=False)["state_dict"]
    unfrozen = torch.load("/ceoas/emapr/sundial/checkpoints/all224_glkn/epoch=0123_val_loss=0.362_fcn_dice_unfrozen.ckpt", map_location=torch.device('cpu'), weights_only=False)["state_dict"]
    nockpt = torch.load("/ceoas/emapr/sundial/checkpoints/all224_glkn/epoch=0127_val_loss=0.414_fcn_dice_nockpt.ckpt", map_location=torch.device('cpu'), weights_only=False)["state_dict"]
    state_dicts = {"No training": random, "Fine Tuned w/ Frozen Encoder": frozen, "Fine Tuned w/ Unfrozen Encoder": unfrozen, "Trained w/o 2.0 Checkpoint": nockpt}

    blocks = {}
    upscalers = {}
    fcns = {}
    for k, sd in state_dicts.items():
        blocks[k] = filter_layers(sd)
        upscalers[k] = filter_layers(sd, "head.0")
        fcns[k] = filter_layers(sd, "head.1")

    l2_distances, l1_distances, cosine_sim = compute_pairwise_distances(blocks)
    plot_distance_matrix(l2_distances, os.path.join(CKPT_COMPARE_PATH, "blocks_l2"), "Pairwise Encoder L2 Distances", blocks.keys())
    plot_distance_matrix(l1_distances, os.path.join(CKPT_COMPARE_PATH, "blocks_l1"), "Pairwise Encoder L1 Distances", blocks.keys())
    plot_distance_matrix(cosine_sim, os.path.join(CKPT_COMPARE_PATH, "blocks_cos"), "Pairwise Encoder Cos Similarity", blocks.keys(), True)

    l2_distances, l1_distances, cosine_sim = compute_pairwise_distances(upscalers)
    plot_distance_matrix(l2_distances, os.path.join(CKPT_COMPARE_PATH, "upscalers_l2"), "Pairwise Upscaler L2 Distances", upscalers.keys())
    plot_distance_matrix(l1_distances, os.path.join(CKPT_COMPARE_PATH, "upscalers_l1"), "Pairwise Upscaler L1 Distances", upscalers.keys())
    plot_distance_matrix(cosine_sim, os.path.join(CKPT_COMPARE_PATH, "upscalers_cos"), "Pairwise Upscaler Cos Similarity", upscalers.keys(), True)

    l2_distances, l1_distances, cosine_sim = compute_pairwise_distances(fcns)
    plot_distance_matrix(l2_distances, os.path.join(CKPT_COMPARE_PATH, "fcns_l2"), "Pairwise FCN Head L2 Distances", fcns.keys())
    plot_distance_matrix(l1_distances, os.path.join(CKPT_COMPARE_PATH, "fcns_l1"), "Pairwise FCN Head L1 Distances", fcns.keys())
    plot_distance_matrix(cosine_sim, os.path.join(CKPT_COMPARE_PATH, "fcns_cos"), "Pairwise FCN Head Cos Similarity", fcns.keys(), True)