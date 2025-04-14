import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations


BASE_PATH = os.getenv("BASE_PATH")
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
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
    cosine_sims = np.ones((num_models, num_models))
    cos = torch.nn.CosineSimilarity(dim=1)
    
    for (a, b), (i, j) in zip(combinations(state_dicts.keys(), 2), combinations(range(num_models), 2)):
        l1_dist = []
        l2_dist = []
        cos_agg = []
        count = 0
        
        parametersA = []
        parametersB = []
        for (name1, param1), (name2, param2) in zip(state_dicts[a].items(), state_dicts[b].items()):
            parametersA.append(param1.flatten())
            parametersB.append(param2.flatten())

        parametersA = result_tensor = torch.concat(parametersA).unsqueeze(0)
        parametersB = result_tensor = torch.concat(parametersB).unsqueeze(0)
        l2_dist = torch.linalg.norm(parametersA-parametersB, ord=2)
        l1_dist = torch.linalg.norm(parametersA-parametersB, ord=1)
        cos_agg = cos(parametersA, parametersB)[0]
        
        l2_distances[i, j] = l2_dist
        l2_distances[j, i] = l2_dist
        l1_distances[i, j] = l1_dist
        l1_distances[j, i] = l1_dist
        cosine_sims[i, j] = cos_agg
        cosine_sims[j, i] = cos_agg
    
    max_l1, min_l1 = l1_distances.max(), l1_distances.min()
    max_l2, min_l2 = l2_distances.max(), l2_distances.min()
    max_cos, min_cos = cosine_sims.max(), cosine_sims.min()
    
    l1_distances = ((l1_distances - min_l1) / (max_l1 - min_l1 + 1e-6))
    l2_distances = ((l1_distances - min_l2) / (max_l2 - min_l2 + 1e-6))
    cosine_sims = ((cosine_sims - min_cos) / (max_cos - min_cos + 1e-6))

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
    # TODO: don't be lazy
    checkpoints = [
        {"state_dict": True, "name": "No training", "path": "/ceoas/emapr/sundial/checkpoints/all224_glkn/random.pt"},
        {"state_dict": False, "name": "Fine Tuned w/ Frozen Encoder", "path": "/ceoas/emapr/sundial/checkpoints/all224_glkn/epoch=0124_val_loss=0.354_fcn_dice_frozen.ckpt"},
        {"state_dict": False, "name": "Fine Tuned w/ Unfrozen Encoder", "path": "/ceoas/emapr/sundial/checkpoints/all224_glkn/epoch=0123_val_loss=0.362_fcn_dice_unfrozen.ckpt"},
        {"state_dict": False, "name": "Trained w/o 2.0 Checkpoint", "path": "/ceoas/emapr/sundial/checkpoints/all224_glkn/epoch=0127_val_loss=0.414_fcn_dice_nockpt.ckpt"},
    ]
    state_dicts = {}
    for p in checkpoints:
        if p["state_dict"]:
           state_dicts[p["name"]] = torch.load(p["path"], map_location=torch.device('cpu'), weights_only=p["state_dict"])
        else:
            state_dicts[p["name"]] = torch.load(p["path"], map_location=torch.device('cpu'), weights_only=p["state_dict"])["state_dict"]

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