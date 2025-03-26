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
                    key,
                    file_path):
    all_data = torch.concat([data[n][key] for n in names], dim=0)
    labels = np.concatenate([data[n]["labels"] for n in names], axis=0)
    
    reducer = TSNE(n_components=2,
                    perplexity=500,
                    metric="euclidean",
                    n_jobs=16,
                    random_state=42,
                    verbose=True)
    fit = reducer.fit(all_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.arange(7))
    
    
    sc = ax.scatter(fit[:, 0], fit[:, 1], c=labels, alpha=0.8, s=1)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=8, label=names[i])
        for i in labels
    ]
    ax.legend(handles=legend_handles, title="Embedding Version", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_ylim(fit.min(), fit.max())
    ax.set_xlim(fit.min(), fit.max())
    
    plt.title(key.replace("_", " "))
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, key), dpi=300)
    plt.close()
    
    return fit


if __name__ == "__main__":
    names = [
        "300m_all_nrs_embed",
        "300m_fcn_dice_nockpt_nrs_embed",
        "300m_fcn_dice_unfrozen_nrs_embed",
        "300m_tl_all_nrs_embed",
        "300m_tl_noboth_nrs_embed",
        "300m_tl_noloca_nrs_embed",
        "300m_tl_notime_nrs_embed"
    ]
    
    data = {}
    for i, exp in enumerate(names):
        data[exp] = {
                "original_t0": torch.load(os.path.join(BASE_PATH, exp, "og_result_t0.pt"), map_location=torch.device('cpu'), weights_only=True),
                "original_t1": torch.load(os.path.join(BASE_PATH, exp, "og_result_t1.pt"), map_location=torch.device('cpu'), weights_only=True),
                "reversed_t0": torch.load(os.path.join(BASE_PATH, exp, "rev_result_t0.pt"), map_location=torch.device('cpu'), weights_only=True),
                "reversed_t1": torch.load(os.path.join(BASE_PATH, exp, "rev_result_t1.pt"), map_location=torch.device('cpu'), weights_only=True),
            }
        data[exp]["labels"] = np.repeat(i, data[exp]["original_t0"].shape[0])
    print("starting original_t0")
    tsne_separation(data,
                    names,
                    "original_t0",
                    os.path.join(BASE_PATH, "all_compare"))
    print("starting original_t1")
    tsne_separation(data,
                    names,
                    "original_t1",
                    os.path.join(BASE_PATH, "all_compare"))
    print("starting reversed_t0")
    tsne_separation(data,
                    names,
                    "reversed_t0",
                    os.path.join(BASE_PATH, "all_compare"))
    print("starting reversed_t1")
    tsne_separation(data,
                    names,
                    "reversed_t1",
                    os.path.join(BASE_PATH, "all_compare"))