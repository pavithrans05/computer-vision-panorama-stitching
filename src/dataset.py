import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class GCPDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):

        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        self.shape_to_idx = {
            "Cross": 0,
            "Square": 1,
            "L-Shape": 2
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.root_dir,
            row["path"]
        )

        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(img_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        h, w = image.shape[:2]

        x_norm = row["x"] / w
        y_norm = row["y"] / h

        keypoints = [(x_norm * w, y_norm * h)]

        if self.transform:

            transformed = self.transform(
                image=image,
                keypoints=keypoints
            )

            image = transformed["image"]

            kp = transformed["keypoints"][0]

            x_norm = kp[0] / image.shape[1]
            y_norm = kp[1] / image.shape[0]

            x_norm = np.clip(x_norm, 0.0, 1.0)
            y_norm = np.clip(y_norm, 0.0, 1.0)

        coords = torch.tensor(
            [x_norm, y_norm],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.shape_to_idx[row["shape"]],
            dtype=torch.long
        )

        image = torch.tensor(
            image.transpose(2, 0, 1),
            dtype=torch.float32
        ) / 255.0

        return {
            "image": image,
            "coords": coords,
            "label": label,
            "path": row["path"]
        }