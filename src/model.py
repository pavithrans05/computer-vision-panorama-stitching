import torch
import torch.nn as nn
import torchvision.models as models


class GCPModel(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feature_dim = 1280

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        coords = self.regressor(x)

        logits = self.classifier(x)

        return coords, logits