"""FairDomain glaucoma dataset: SLO + OCT fundus (224x224 uint8) -> glaucoma (0/1)."""
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = glob.glob("/home/rushi/dm/extracted/*/fairdomain")[0]


class FairDomainGlaucoma(Dataset):
    """Loads every sample into RAM (~500 MB total) for fast epochs.

    Returns (image[2,224,224] float in [0,1], label, attrs dict-free tensor index).
    Channel 0 = SLO, channel 1 = OCT. Group attributes (race/gender/ethnicity)
    are kept as string arrays in `self.meta` for fairness evaluation only.
    """

    def __init__(self, split="training", augment=False, root=ROOT):
        assert split in ("training", "testing")
        df = pd.read_csv(os.path.join(root, "data_summary.csv"))
        df = df[df["use"] == split].reset_index(drop=True)
        folder = "Training" if split == "training" else "Testing"

        slo, oct_ = [], []
        for fn in df["filename"]:
            d = np.load(os.path.join(root, folder, fn))
            slo.append(d["slo_fundus"])
            oct_.append(d["oct_fundus"])
        self.x = torch.from_numpy(np.stack([np.stack(slo), np.stack(oct_)], axis=1))  # N,2,H,W uint8
        self.y = torch.from_numpy((df["glaucoma"] == "yes").astype(np.int64).values)
        self.meta = df[["age", "gender", "race", "ethnicity", "md"]].copy()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].float() / 255.0
        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            # small brightness/contrast jitter, applied per modality
            x = x * (1 + 0.1 * (torch.rand(2, 1, 1) - 0.5) * 2) + 0.05 * (torch.rand(2, 1, 1) - 0.5)
            x = x.clamp(0, 1)
        return x, self.y[i]
