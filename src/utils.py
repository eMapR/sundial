import glob
import os
import re


def get_best_ckpt(dir_path):
    pattern = "epoch-*_val_loss-*.ckpt"
    regex = re.compile(r"epoch-(\d+)_val_loss-(\d+\.\d+)(?:-v(\d+))?\.ckpt")

    files = glob.glob(os.path.join(dir_path, pattern))

    min_val_loss = float('inf')
    current_epoch = -1
    current_version = -1
    best_file = None

    for file in files:
        match = regex.search(file)
        if match:
            epoch, val_loss, version = match.groups()
            if version is None:
                version = -2
            else:
                version = int(version)
            epoch = int(epoch)
            val_loss = float(val_loss)

            if val_loss < min_val_loss or (val_loss == min_val_loss and (epoch > current_epoch or (epoch == current_epoch and version > current_version))):
                min_val_loss = val_loss
                current_epoch = int(epoch)
                current_version = version
                best_file = file
    if best_file is not None:
        return best_file
    else:
        raise FileNotFoundError("No checkpoint found in the directory.")
