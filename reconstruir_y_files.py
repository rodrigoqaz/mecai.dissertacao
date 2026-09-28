import os
import numpy as np
from src.data.data_loader import NPYFolderDataset

VERSIONS = ["v1", "v9", "v10"]
MODEL = "convnext"


def rebuild(version, model=MODEL):
    ds = NPYFolderDataset(f"data/gold/datasets/{version}/test")
    paths = [p for p, _ in ds.samples]
    labels = np.array([l for _, l in ds.samples])
    f = f"results/predictions/{model}_{version}_preds.npz"
    d = dict(np.load(f, allow_pickle=True))
    assert len(paths) == len(d["y_true"]), (len(paths), len(d["y_true"]))
    assert (labels == d["y_true"]).all(), f"ordem do dataset != ordem do .npz para {version}"
    d["y_files"] = np.array([os.path.basename(p) for p in paths])
    np.savez_compressed(f, **d)
    print(f"[OK] {version}: y_files anexado ({len(paths)} patches)")


if __name__ == "__main__":
    for v in VERSIONS:
        rebuild(v)
