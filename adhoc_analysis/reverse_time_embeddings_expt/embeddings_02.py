import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde, linregress
from matplotlib.colors import LinearSegmentedColormap
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
EXPERIMENT_SUFFIX = os.getenv("EXPERIMENT_SUFFIX")
EXPERIMENT_PATH = os.path.join(BASE_PATH, EXPERIMENT_SUFFIX)


def plot_elementwise_distances(A,
                               B,
                               labels,
                               x_labels,
                               y_label,
                               file_name,
                               max_l2=None,
                               min_l2=None,
                               max_l1=None,
                               min_l1=None,
                               max_cos=None,
                               min_cos=None):
    distances = A - B
    l2_distance = torch.linalg.norm(distances, ord=2, dim=1)
    l1_distance = torch.linalg.norm(distances, ord=1, dim=1)
    cos = torch.nn.CosineSimilarity(dim=1)
    cosine_sim = cos(A, B)

    if max_l2 is None or min_l2 is None:
        max_l2, min_l2 = l2_distance.max().item(), l2_distance.min().item()
    if max_l1 is None or min_l1 is None:
        max_l1, min_l1 = l1_distance.max().item(), l1_distance.min().item()
    if max_cos is None or min_cos is None:
        max_cos, min_cos = cosine_sim.max().item(), cosine_sim.min().item()
    
    l2_maxmin = max_l2 - min_l2
    l1_maxmin = max_l1 - min_l1
    cos_maxmin = max_cos - min_cos

    l2_distance_np = (l2_distance.cpu().numpy() - min_l2) / l2_maxmin
    l1_distance_np = (l1_distance.cpu().numpy() - min_l1) / l1_maxmin
    cosine_sim_np = (cosine_sim.cpu().numpy() - min_cos) / cos_maxmin

    def plot_with_regression(ax, x, y, title, x_label, y_label, ylim):
        ax.scatter(x, y, alpha=0.7, s=1)

        if "NDVI" not in x_label:
            slope, intercept = np.polyfit(x, y, 1)
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = slope * x_fit + intercept
            
            y_pred = slope * x + intercept
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            ax.plot(x_fit, y_fit, color='red', linewidth=2, label=f'Regression: y={slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.3f}')
            
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(ylim)
        ax.legend()

    for label, x_label in zip(labels, x_labels):
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        labels_np = label.numpy() if isinstance(label, torch.Tensor) else label

        print(l1_distance_np.shape, l2_distance_np.shape, labels_np.shape, max_l2, min_l2, max_l1, min_l1)
        plot_with_regression(axes[0], labels_np, l2_distance_np, "Absolute L2 Distance", x_label, y_label, (0, 1))
        plot_with_regression(axes[1], labels_np, l1_distance_np, "Absolute L1 Distance", x_label, y_label, (0, 1))
        plot_with_regression(axes[2], labels_np, cosine_sim_np, "Cosine Similarity", x_label, "Cosine Similarity", (0, 1))
        
        plt.tight_layout()
        plt.savefig(f"{file_name}_{x_label.lower()}", dpi=300)
        plt.close() 
    return max_l2, min_l2, max_l1, min_l1, max_cos, min_cos


def tsne_separation(data,
                    labels,
                    labels_suffixes,
                    file_path,
                    s0=.5,
                    s1=.5,
                    label0='Timestep 0',
                    label1='Timestep 1',
                    reducer = None):
    print("Data set contains %d samples with %d features" % data.shape)
    if reducer is None:
        reducer = TSNE(n_components=2,
                    perplexity=500,
                    metric="euclidean",
                    n_jobs=16,
                    random_state=42,
                    verbose=True)
        fit = reducer.fit(data)
    else:
        fit = reducer.transform(data)
    
    for label, label_suf in zip(labels, labels_suffixes):
        plt.figure(figsize=(10, 6))
        
        color_0 = LinearSegmentedColormap.from_list("0", ["grey", "lightblue", "blue", "darkblue"])
        color_1 = LinearSegmentedColormap.from_list("1", ["grey", "lightcoral", "red", "darkred"])
        num_patches = int(data.shape[0] // 2)

        sc0 = plt.scatter(fit[:num_patches, 0], fit[:num_patches, 1],
                        c=label, cmap=color_0,
                        alpha=0.8, s=s0, label=label0)
        sc1 = plt.scatter(fit[num_patches:num_patches*2, 0], fit[num_patches:num_patches*2, 1],
                        c=label, cmap=color_1,
                        alpha=0.8, s=s1, label=label1)
        
        cb0 = plt.colorbar(sc0,  fraction=0.06)
        cb0.set_label(f'{label0} {label_suf}')
        cb1 = plt.colorbar(sc1,  fraction=0.06)
        cb1.set_label(f'{label1} {label_suf}')

        plt.ylim(fit.min(), fit.max())
        plt.xlim(fit.min(), fit.max())
        plt.tight_layout()
        plt.savefig(f"{file_path}_{label_suf.lower().replace(' ','_')}", dpi=300)
        plt.close()
    
    return fit

 
def plot_max_diff_embeddings(embeddings,
                        labels,
                        labels_suffixes,
                        file_path,
                        ylim):
    N, D = embeddings.shape
    
    x_vals = np.tile(np.arange(D), N)
    y_vals = embeddings.numpy().flatten()
    
    for label, label_suf in zip(labels, labels_suffixes):
        colors = np.repeat(label.numpy(), D) 

        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(x_vals, y_vals, c=colors, alpha=0.75, s=.1)
        plt.colorbar(scatter, label="Class")
        plt.ylim(-ylim, ylim)
        
        plt.xlabel("Embedding Index (0-1023)")
        plt.ylabel("Embedding Difference")
        plt.title(f"Feature Distribution Colored by {label_suf}")
        plt.savefig(f"{file_path}_{label_suf}", dpi=300)


if __name__ == "__main__":
    og_t0_embed = torch.load(os.path.join(EXPERIMENT_PATH, "og_result_t0.pt"), map_location=torch.device('cpu'), weights_only=True)
    og_t1_embed = torch.load(os.path.join(EXPERIMENT_PATH, "og_result_t1.pt"), map_location=torch.device('cpu'), weights_only=True)

    rev_t0_embed = torch.load(os.path.join(EXPERIMENT_PATH, "rev_result_t0.pt"), map_location=torch.device('cpu'), weights_only=True)
    rev_t1_embed = torch.load(os.path.join(EXPERIMENT_PATH, "rev_result_t1.pt"), map_location=torch.device('cpu'), weights_only=True)

    band_t0 = torch.load(os.path.join(EXPERIMENT_PATH, "band_t0.pt"), map_location=torch.device('cpu'), weights_only=True)
    band_t1 = torch.load(os.path.join(EXPERIMENT_PATH, "band_t1.pt"), map_location=torch.device('cpu'), weights_only=True)
    
    ndvi_t0 = torch.load(os.path.join(EXPERIMENT_PATH, "ndvi_t0.pt"), map_location=torch.device('cpu'), weights_only=True)
    ndvi_t1 = torch.load(os.path.join(EXPERIMENT_PATH, "ndvi_t1.pt"), map_location=torch.device('cpu'), weights_only=True)
    
    og_sum_classes = torch.load(os.path.join(EXPERIMENT_PATH, "og_classes.pt"), map_location=torch.device('cpu'), weights_only=True)
    rev_sum_classes = torch.load(os.path.join(EXPERIMENT_PATH, "rev_classes.pt"), map_location=torch.device('cpu'), weights_only=True)

    assert len(rev_t0_embed) == len(rev_t1_embed)
    assert len(rev_t0_embed) == len(og_t0_embed)
    assert len(og_t0_embed) == len(og_t1_embed)
    assert len(rev_sum_classes) == len(og_sum_classes)
    assert (rev_sum_classes-og_sum_classes).sum() == 0
    
    ndvi_diff = ndvi_t1 - ndvi_t0
    ndvi_diff = torch.mean(ndvi_diff, dim=1)
    mask = torch.isnan(ndvi_diff)
    
    # lets get rid of those pesky nans and inf
    ndvi_diff = ndvi_diff[~mask]
    og_t0_embed = og_t0_embed[~mask]
    og_t1_embed = og_t1_embed[~mask]
    
    rev_t0_embed = rev_t0_embed[~mask]
    rev_t1_embed = rev_t1_embed[~mask]
    
    band_t0 = band_t0[~mask]
    band_t1 = band_t1[~mask]

    ndvi_t0 = ndvi_t0[~mask]
    ndvi_t1 = ndvi_t1[~mask]
    
    og_sum_classes = og_sum_classes[~mask]
    rev_sum_classes = rev_sum_classes[~mask]
    
    print("how does order affect the embeddings all at once")
    plot_elementwise_distances(torch.concat([og_t0_embed, og_t1_embed], dim=1),
                               torch.concat([rev_t0_embed, rev_t1_embed], dim=1),
                               [og_sum_classes, ndvi_diff],
                               ["# px of Δ", "NDVI Δ T1-T0"],
                               "distance w/ 2 time steps",
                               os.path.join(EXPERIMENT_PATH, "dist_all_embed_all"))

    print("how does the order affect the embeddings as a whole")
    max_l2, min_l2, max_l1, min_l1, max_cos, min_cos = plot_elementwise_distances(og_t0_embed,
                                            rev_t0_embed,
                                            [og_sum_classes, ndvi_diff],
                                            ["# px of Δ", "NDVI Δ T1-T0"],
                                            "distance w/ 1 time step (t = 0)",
                                            os.path.join(EXPERIMENT_PATH,"dist_all_embed_t0"))
    _, _, _, _, _, _ = plot_elementwise_distances(og_t1_embed,
                                            rev_t1_embed,
                                            [og_sum_classes, ndvi_diff],
                                            ["# px of Δ", "NDVI Δ T1-T0"],
                                            "distance w/ 1 time step (t = 1)",
                                            os.path.join(EXPERIMENT_PATH, "dist_all_embed_t1"),
                                            max_l2, min_l2, max_l1, min_l1, max_cos, min_cos)
    print(max_l2, min_l2, max_l1, min_l1)

    print("how does the order affect the embeddings on the w pos embed")
    max_l2, min_l2, max_l1, min_l1, max_cos, min_cos = plot_elementwise_distances(og_t0_embed[:,HW_START_IDX:HW_END_IDX],
                                            rev_t0_embed[:,HW_START_IDX:HW_END_IDX],
                                            [og_sum_classes, ndvi_diff],
                                            ["# px of Δ", "NDVI Δ T1-T0"],
                                            "distance w/ 1 time step (t = 0)",
                                            os.path.join(EXPERIMENT_PATH, "dist_hw_embed_t0"))
    _, _, _, _, _, _ = plot_elementwise_distances(og_t1_embed[:,HW_START_IDX:HW_END_IDX],
                                            rev_t1_embed[:,HW_START_IDX:HW_END_IDX],
                                            [og_sum_classes, ndvi_diff],
                                            ["# px of Δ", "NDVI Δ T1-T0"],
                                            "distance w/ 1 time step (t = 1)",
                                            os.path.join(EXPERIMENT_PATH, "dist_hw_embed_t1"),
                                            max_l2, min_l2, max_l1, min_l1, max_cos, min_cos)

    print("how does the order affect the embeddings on the t pos embed")
    max_l2, min_l2, max_l1, min_l1, max_cos, min_cos = plot_elementwise_distances(og_t0_embed[:,T_START_IDX:T_END_IDX],
                                            rev_t0_embed[:,T_START_IDX:T_END_IDX],
                                            [og_sum_classes, ndvi_diff],
                                            ["# px of Δ", "NDVI Δ T1-T0"],
                                            "distance w/ 1 time step (t = 0)",
                                            os.path.join(EXPERIMENT_PATH, "dist_t_embed_t0"))
    _, _, _, _, _, _ = plot_elementwise_distances(og_t1_embed[:,T_START_IDX:T_END_IDX],
                                            rev_t1_embed[:,T_START_IDX:T_END_IDX],
                                            [og_sum_classes, ndvi_diff],
                                            ["# px of Δ", "NDVI Δ T1-T0"],
                                            "distance w/ 1 time step (t = 1)",
                                            os.path.join(EXPERIMENT_PATH, "dist_t_embed_t1"),
                                            max_l2, min_l2, max_l1, min_l1, max_cos, min_cos)

    print("how separation look like with just band values")
    tsne_separation(torch.concat([band_t0, band_t1],dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_bands"))

    print("how separation look like with just ndvi values")
    tsne_separation(torch.concat([ndvi_t0, ndvi_t1],dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_ndvi"))


    print("tsne separations for both years (original)")
    oga = tsne_separation(torch.concat([og_t0_embed,
                                        og_t1_embed], dim=0),
                          [og_sum_classes, ndvi_diff],
                          ["# px of Δ", "NDVI Δ T1-T0"],
                          os.path.join(EXPERIMENT_PATH,"tsne_all_embed_og"))
    oghw = tsne_separation(torch.concat([og_t0_embed[:,HW_START_IDX:HW_END_IDX],
                                         og_t1_embed[:,HW_START_IDX:HW_END_IDX]], dim=0),
                           [og_sum_classes, ndvi_diff],
                           ["# px of Δ", "NDVI Δ T1-T0"],
                           os.path.join(EXPERIMENT_PATH, "tsne_hw_embed_og"))
    ogt = tsne_separation(torch.concat([og_t0_embed[:,T_START_IDX:T_END_IDX],
                                        og_t1_embed[:,T_START_IDX:T_END_IDX]], dim=0),
                          [og_sum_classes, ndvi_diff],
                          ["# px of Δ", "NDVI Δ T1-T0"],
                          os.path.join(EXPERIMENT_PATH, "tsne_t_embed_og"))
    
    print("tsne separations for both years (reverse)")
    tsne_separation(torch.concat([rev_t0_embed,
                                  rev_t1_embed], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_all_embed_rev_ogfit"),
                    reducer=oga)
    tsne_separation(torch.concat([rev_t0_embed[:,HW_START_IDX:HW_END_IDX],
                                  rev_t1_embed[:,HW_START_IDX:HW_END_IDX]], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_hw_embed_rev_ogfit"),
                    reducer=oghw)
    tsne_separation(torch.concat([rev_t0_embed[:,T_START_IDX:T_END_IDX],
                                  rev_t1_embed[:,T_START_IDX:T_END_IDX]], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_t_embed_rev_ogfit"),
                    reducer=ogt)

    print("tsne separations for same year")
    tsne_separation(torch.concat([og_t0_embed,
                                  rev_t0_embed], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_all_embed_t0_og_rev"),
                    label0='T0 Original',
                    label1='T0 Reversed',
                    s0=1.5)
    tsne_separation(torch.concat([og_t1_embed,
                                  rev_t1_embed], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH,"tsne_all_embed_t1_og_rev"),
                    label0='T1 Original',
                    label1='T1 Reversed',
                    s0=1.5)

    tsne_separation(torch.concat([og_t0_embed[:,HW_START_IDX:HW_END_IDX],
                                  rev_t0_embed[:,HW_START_IDX:HW_END_IDX]], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_hw_embed_t0_og_rev"),
                    label0='T0 Original',
                    label1='T0 Reversed',
                    s0=1.5)
    tsne_separation(torch.concat([og_t1_embed[:,HW_START_IDX:HW_END_IDX],
                                  rev_t1_embed[:,HW_START_IDX:HW_END_IDX]], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_hw_embed_t1_og_rev"),
                    label0='T1 Original',
                    label1='T1 Reversed',
                    s0=1.5)

    tsne_separation(torch.concat([og_t0_embed[:,T_START_IDX:T_END_IDX],
                                  rev_t0_embed[:,T_START_IDX:T_END_IDX]], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_t_embed_t0_og_rev"),
                    label0='T0 Original',
                    label1='T0 Reversed',
                    s0=1.5)
    tsne_separation(torch.concat([og_t1_embed[:,T_START_IDX:T_END_IDX],
                                  rev_t1_embed[:,T_START_IDX:T_END_IDX]], dim=0),
                    [og_sum_classes, ndvi_diff],
                    ["# px of Δ", "NDVI Δ T1-T0"],
                    os.path.join(EXPERIMENT_PATH, "tsne_t_embed_t1_og_rev"),
                    label0='T1 Original',
                    label1='T1 Reversed',
                    s0=1.5)

    # I should probably rewrite this into a function min max normalization
    og_embed_diff = og_t1_embed - og_t0_embed
    og_min = og_embed_diff.min()
    og_max = og_embed_diff.max()
    og_embed_diff = (og_embed_diff - og_min) / og_max - og_min
    
    rev_embed_diff = rev_t1_embed - rev_t0_embed
    rev_embed_diff = (rev_embed_diff - og_min) / og_max - og_min
    
    t0_embed_diff = og_t0_embed - rev_t0_embed
    t0_min = t0_embed_diff.min()
    t0_max = t0_embed_diff.max()
    t0_embed_diff = (t0_embed_diff - t0_min) / t0_max - t0_min
    
    t1_embed_diff = og_t1_embed - rev_t1_embed
    t1_embed_diff = (t1_embed_diff - t0_min) / t0_max - t0_min

    max_diff = og_embed_diff - rev_embed_diff
    

    plot_max_diff_embeddings(max_diff,
                            [og_sum_classes, ndvi_diff],
                            ["# px of Δ", "NDVI Δ T1-T0"],
                            os.path.join(EXPERIMENT_PATH, "max_diff_second_order"),
                            og_max)
    plot_max_diff_embeddings(og_embed_diff,
                            [og_sum_classes, ndvi_diff],
                            ["# px of Δ", "NDVI Δ T1-T0"],
                            os.path.join(EXPERIMENT_PATH, "max_diff_og"),
                            og_max)
    plot_max_diff_embeddings(rev_embed_diff,
                            [og_sum_classes, ndvi_diff],
                            ["# px of Δ", "NDVI Δ T1-T0"],
                            os.path.join(EXPERIMENT_PATH, "max_diff_rev"),
                            og_max)
    plot_max_diff_embeddings(t0_embed_diff,
                            [og_sum_classes, ndvi_diff],
                            ["# px of Δ", "NDVI Δ T1-T0"],
                            os.path.join(EXPERIMENT_PATH, "t0_embed_diff"),
                            og_max)
    plot_max_diff_embeddings(t1_embed_diff,
                            [og_sum_classes, ndvi_diff],
                            ["# px of Δ", "NDVI Δ T1-T0"],
                            os.path.join(EXPERIMENT_PATH, "t1_embed_diff"),
                            og_max)
