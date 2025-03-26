import torch
import numpy as np
import os


EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
EXPERIMENT_SUFFIX = os.getenv("EXPERIMENT_SUFFIX")
BASE_PATH = os.getenv("BASE_PATH")
EXPERIMENT_PATH = os.path.join(BASE_PATH, EXPERIMENT_SUFFIX)
NUM_FRAMES = 2

def load_embeddings_and_annotations(og_path, rev_path, embed_fstr, anno_fstr, indx, class_indx):
    selected_t0, selected_t1, selected_t0_rev, selected_t1_rev = [], [], [], []
    og_classes, rev_classes = [], []
    
    for img_indx, time_indx in indx:
        img, time = int(img_indx), int(time_indx)
        
        embed_path = embed_fstr.format(img_indx=img, time_indx=time)
        anno_path = anno_fstr.format(img_indx=img, time_indx=time)
        
        embed = torch.load(os.path.join(og_path, embed_path), map_location=torch.device('cpu'))
        embed_rev = torch.load(os.path.join(rev_path, embed_path), map_location=torch.device('cpu'))
        anno = torch.load(os.path.join(og_path, anno_path), map_location=torch.device('cpu'))[class_indx]
        anno_rev = torch.load(os.path.join(rev_path, anno_path), map_location=torch.device('cpu'))[class_indx]
        
        condition = (anno > 0)
        condition_rev = (anno_rev > 0)
        
        ann_indices = torch.where(condition)[0]
        ann_indices_rev = torch.where(condition_rev)[0]
        
        if ann_indices.numel() == 0:
            continue

        og_classes.append(anno[ann_indices])
        rev_classes.append(anno_rev[ann_indices_rev])

        print(f"processing tensor {img} of shape {embed.shape}")
        if len(embed.shape) == 2:
            if embed.shape[0] > 392:
                embed = embed[1:] # remove cls token
            if embed_rev.shape[0] > 392:
                embed_rev = embed_rev[1:] # remove cls token
            
            L, D = embed.shape
            L = L // NUM_FRAMES
            P = int(L ** 0.5)
            
            embed = embed.transpose(1,0)
            embed_rev = embed_rev.transpose(1,0)
        else:
            # should probably parameterize this
            D, P, P = embed.shape
            D = D // NUM_FRAMES

        embed = embed.reshape(D, NUM_FRAMES, P*P)
        embed_rev = embed_rev.reshape(D, NUM_FRAMES, P*P)
            
        selected_t0.append(embed[:, 0, ann_indices_rev].permute(1,0))
        selected_t1.append(embed[:, 1, ann_indices_rev].permute(1,0))
        
        # index the time different since they are reversed
        selected_t0_rev.append(embed_rev[:, 1, ann_indices_rev].permute(1,0))
        selected_t1_rev.append(embed_rev[:, 0, ann_indices_rev].permute(1,0))
    
    return selected_t0, selected_t1, selected_t0_rev, selected_t1_rev, og_classes, rev_classes


def save_results(selected_t0, selected_t1, selected_t0_rev, selected_t1_rev, og_classes, rev_classes):
    torch.save(torch.concat(og_classes, dim=0), os.path.join(EXPERIMENT_PATH, "og_classes.pt"))
    torch.save(torch.concat(rev_classes, dim=0), os.path.join(EXPERIMENT_PATH, "rev_classes.pt"))
    torch.save(torch.concat(selected_t0, dim=0), os.path.join(EXPERIMENT_PATH, "og_result_t0.pt"))
    torch.save(torch.concat(selected_t1, dim=0), os.path.join(EXPERIMENT_PATH, "og_result_t1.pt"))
    torch.save(torch.concat(selected_t0_rev, dim=0), os.path.join(EXPERIMENT_PATH, "rev_result_t0.pt"))
    torch.save(torch.concat(selected_t1_rev, dim=0), os.path.join(EXPERIMENT_PATH, "rev_result_t1.pt"))


def main():
    indx = np.load(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/train_sample.npy")
    og_path = os.path.join("/home/ceoas/truongmy/emapr/sundial/predictions/", EXPERIMENT_NAME, EXPERIMENT_SUFFIX)
    rev_path = os.path.join("/home/ceoas/truongmy/emapr/sundial/predictions/", EXPERIMENT_NAME, EXPERIMENT_SUFFIX + "_rev")
    
    embed_fstr = "{img_indx:08d}_t{time_indx:02d}_embed.pt"
    anno_fstr = "{img_indx:08d}_t{time_indx:02d}_anno.pt"
    class_indx = 0
    
    selected_t0, selected_t1, selected_t0_rev, selected_t1_rev, og_classes, rev_classes = \
        load_embeddings_and_annotations(og_path, rev_path, embed_fstr, anno_fstr, indx, class_indx)
    
    save_results(selected_t0, selected_t1, selected_t0_rev, selected_t1_rev, og_classes, rev_classes)


if __name__ == "__main__":
    main()