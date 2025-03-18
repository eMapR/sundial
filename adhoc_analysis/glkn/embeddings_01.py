import torch
import numpy as np
import os


EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
EXPERIMENT_SUFFIX = os.getenv("EXPERIMENT_SUFFIX")
BASE_PATH = os.getenv("BASE_PATH")
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)
EXPERIMENT_PATH = os.path.join(BASE_PATH, EXPERIMENT_SUFFIX)
if not os.path.exists(EXPERIMENT_PATH):
    os.mkdir(EXPERIMENT_PATH)

indx = np.load(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/train_sample.npy")
og_path = os.path.join("/home/ceoas/truongmy/emapr/sundial/predictions/", EXPERIMENT_NAME, EXPERIMENT_SUFFIX)
rev_path = os.path.join("/home/ceoas/truongmy/emapr/sundial/predictions/", EXPERIMENT_NAME, EXPERIMENT_SUFFIX+"_rev")
embed_fstr = "{img_indx:08d}_t{time_indx:02d}_embed_wcls.pt"
anno_fstr = "{img_indx:08d}_t{time_indx:02d}_sumpool_anno.pt"
class_indx = 0

selected_t0 = []
selected_t1 = []
selected_t0_rev = []
selected_t1_rev = []
og_classes = []
rev_classes = []
vector_size = 1024

for img_indx, time_indx in indx:
    img = int(img_indx)
    time = int(time_indx)

    embed_path = embed_fstr.format(img_indx=img, time_indx=time)
    anno_path = anno_fstr.format(img_indx=img, time_indx=time)
    
    embed = torch.load(os.path.join(og_path, embed_path), map_location=torch.device('cpu'), weights_only=True)
    if embed.shape[0] > 392:
        embed = embed[1:] # remove cls token
    embed_rev = torch.load(os.path.join(rev_path, embed_path), map_location=torch.device('cpu'), weights_only=True)
    if embed_rev.shape[0] > 392:
        embed_rev = embed_rev[1:] # remove cls token
    anno = torch.load(os.path.join(og_path, anno_path), map_location=torch.device('cpu'), weights_only=True)[class_indx]
    anno_rev = torch.load(os.path.join(rev_path, anno_path), map_location=torch.device('cpu'), weights_only=True)[class_indx]

    L, D = embed.shape
    L = int(L/2)
    condition = (anno > 0)
    condition_rev = (anno_rev > 0)
    
    ann_indices = torch.where(condition)[0]
    ann_indices_rev = torch.where(condition_rev)[0]
    if ann_indices.numel() == 0:
        continue 
    else:
        og_classes.append(anno[ann_indices])
        rev_classes.append(anno_rev[ann_indices_rev])
        
        selected_t0.append(embed[:L][ann_indices])
        selected_t1.append(embed[L:][ann_indices])
        selected_t0_rev.append(embed_rev[L:][ann_indices_rev])
        selected_t1_rev.append(embed_rev[:L][ann_indices_rev])

if selected_t0:
    og_classes = torch.concat(og_classes, dim=0)
    rev_classes = torch.concat(rev_classes, dim=0)
    result_t0 = torch.concat(selected_t0, dim=0)
    result_t1 = torch.concat(selected_t1, dim=0)
    result_t0_rev = torch.concat(selected_t0_rev, dim=0)
    result_t1_rev = torch.concat(selected_t1_rev, dim=0)
    
    torch.save(og_classes, os.path.join(EXPERIMENT_PATH, "glkn_og_classes.pt"))
    torch.save(rev_classes, os.path.join(EXPERIMENT_PATH, "glkn_rev_classes.pt"))
    torch.save(result_t0, os.path.join(EXPERIMENT_PATH, "glkn_og_result_t0.pt"))
    torch.save(result_t1, os.path.join(EXPERIMENT_PATH, "glkn_og_result_t1.pt"))
    torch.save(result_t0_rev, os.path.join(EXPERIMENT_PATH, "glkn_rev_result_t0.pt"))
    torch.save(result_t1_rev, os.path.join(EXPERIMENT_PATH, "glkn_rev_result_t1.pt"))