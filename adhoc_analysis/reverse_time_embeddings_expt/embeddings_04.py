import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score
from skimage.metrics import structural_similarity as ssim


CLASS_IDX = 6
H = W = 224
TIME_STEP = 1
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
BASE_PATH = os.getenv("BASE_PATH")
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
SAVE_PATH = os.path.join(BASE_PATH, "decision_trees")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def load_sample(chip_data: xr.DataArray,
                anno_data: xr.DataArray,
                samples,
                reverse=False):
    x = []
    y = []
    count = 0
    
    for sample in samples:
        samp_idx = int(sample[1])
        anno_idx = samp_idx - TIME_STEP
        
        count += 1
        chip = chip_data.sel(sample=sample[0]).isel({'datetime': slice(samp_idx-TIME_STEP, samp_idx+1)}).values
        anno = anno_data.sel(sample=sample[0]).isel({'datetime': anno_idx, 'class': CLASS_IDX}).values
        if reverse:
            chip = np.flip(chip, axis=1)
        C, T, H, W = chip.shape
        x.append(chip.transpose(2, 3, 0, 1).reshape(-1, C * T))
        y.append(anno.reshape(-1))
    
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    print(f"flattened {count} images")

    return x, y, count

def run_random_forest(x, y):
    clf = RandomForestClassifier(n_estimators=100,
                                 max_depth=16,
                                 random_state=42,
                                 n_jobs=-1,
                                 verbose=1)
    clf.fit(x, y)
    
    with open(os.path.join(SAVE_PATH, 'random_forest_clf.pkl'), 'wb') as f:
        pickle.dump(clf, f)
        
    return clf

def pred_to_imgs(pred, samples, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for p, sample in zip(pred, samples):
        name = str(int(sample[0])).zfill(8)
        samp_idx = int(sample[1])
        plt.imshow(p, cmap="grey")
        plt.savefig(os.path.join(path, f'{name}-{samp_idx}_pred.png'), dpi=300)
        plt.close()

def predict_n_score(clf, data, labels, count, samples, clf_name, save=False):
    pred_flat = clf.predict(data)
    print(pred_flat.shape)
    pred = pred_flat.reshape(count, H, W)

    jaccard = jaccard_score(pred_flat, labels)
    accuracy = accuracy_score(pred_flat, labels)
    precision = precision_score(pred_flat, labels)
    recall = recall_score(pred_flat, labels)
    
    labels_unflat = labels.reshape(count, H, W)
    ssim_scores = [ssim(pred[i], labels_unflat[i], data_range=1) for i in range(count)]
    mean_ssim = np.mean(ssim_scores)

    print(f"{clf_name} Jaccard Index (IoU): {jaccard:.4f}")
    print(f"{clf_name} Accuracy: {accuracy:.4f}")
    print(f"{clf_name} Precision: {precision:.4f}")
    print(f"{clf_name} Recall: {recall:.4f}")
    print(f"{clf_name} SSIM: {mean_ssim:.4f}")
    np.save(os.path.join(SAVE_PATH, f'{clf_name}.npy'), pred)
    if save:
        pred_to_imgs(pred, samples, os.path.join(SAVE_PATH, clf_name))


if __name__ == "__main__":
    chip_data = xr.open_dataarray(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/chip_data", engine="zarr")
    anno_data = xr.open_dataarray(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/anno_data", , engine="zarr")
    train = np.load(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/train_sample.npy")
    
    print("loading data")
    data, labels, _ = load_sample(chip_data,
                                anno_data,
                                train)
    print("training rf")
    rf = run_random_forest(data, labels)
    
    validate = np.load(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/validate_sample.npy")
    data, labels, count = load_sample(chip_data,
                                    anno_data,
                                    validate)
    
    predict_n_score(rf, data, labels, count, validate, "random_forest")
    
    test = np.load(f"/home/ceoas/truongmy/emapr/sundial/samples/{EXPERIMENT_NAME}/test_sample.npy")
    data, labels, count = load_sample(chip_data,
                                    anno_data,
                                    test)
    predict_n_score(rf, data, labels, count, test, "random_forest", True)
    
    data, labels, count = load_sample(chip_data,
                                    anno_data,
                                    test,
                                    True)
    predict_n_score(rf, data, labels, count, test, "random_forest_rev", True)
    

