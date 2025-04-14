import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from openTSNE import TSNE


EMBED_DIM = 1024
W_EMBED_DIM = EMBED_DIM // 16 * 6
H_EMBED_DIM = EMBED_DIM // 16 * 6
T_EMBED_DIM = EMBED_DIM // 16 * 4

HW_START_IDX = 0
HW_END_IDX = W_EMBED_DIM + H_EMBED_DIM
T_START_IDX = HW_END_IDX
T_END_IDX = EMBED_DIM

BASE_PATH = os.getenv("BASE_PATH")


def tsne_separation(data,
                    names,
                    file_path):
    all_data = torch.concat([data[n]["data"] for n in names], dim=0)
    
    reducer = TSNE(n_components=2,
                    perplexity=30,
                    metric="euclidean",
                    n_jobs=16,
                    random_state=42,
                    verbose=True)
    fit = reducer.fit(all_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    
    num_patches = len(data[names[0]]["data"])
    start = 0
    legend_handles = []
    s = len(names) + 1

    for i in range(len(names)):
        _ = ax.scatter(fit[start:start+num_patches, 0], fit[start:start+num_patches, 1], c=[cmap(i)], alpha=0.8, s=s)
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=8, label=names[i]))
        s -= 1
        start += num_patches

    ax.legend(handles=legend_handles, title="Embedding Version", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_ylim(fit.min(), fit.max())
    ax.set_xlim(fit.min(), fit.max())
    
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, "original"), dpi=300)
    plt.close()
    
    return fit


if __name__ == "__main__":
    names = [
        # "300m_all_nrs_embed",
        # "300m_fcn_dice_nockpt_nrs_embed",
        # "300m_fcn_dice_unfrozen_nrs_embed",
        "300m_tl_all_nrs_embed",
        "300m_tl_noboth_nrs_embed",
        "300m_tl_noloca_nrs_embed",
        "300m_tl_notime_nrs_embed",
    ]
    
    data = {}
    for i, exp in enumerate(names):
        t0 = torch.load(os.path.join(BASE_PATH, exp, "og_result_t0.pt"), map_location=torch.device('cpu'), weights_only=True)
        t1 = torch.load(os.path.join(BASE_PATH, exp, "og_result_t1.pt"), map_location=torch.device('cpu'), weights_only=True)
        data[exp] = {'data': torch.concat([t0,t1], dim=1)}

    tsne_separation(data,
                    names,
                    os.path.join(BASE_PATH, "all_compare"))
    