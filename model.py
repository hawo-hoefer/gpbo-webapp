import torch
from torch import nn
from torch import Tensor


class ScaleConstant(nn.Module):
    def __init__(self):
        super().__init__()

        self.f = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, X: Tensor) -> Tensor:
        return self.f * X


class XRDClassifier(nn.Module):
    def __init__(self, in_size: int = 2048, ch: int = 32, ks: int = 13) -> None:
        super().__init__()

        def block(in_ch, out_ch, ks, pool, dropout=True):
            return nn.Sequential(
                nn.BatchNorm1d(in_ch),
                *([nn.Dropout1d(p=0.2)] if dropout else []),
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.LeakyReLU(),
                nn.AvgPool1d(pool)
            )

        self.conv = nn.Sequential(
            block(1, ch, ks, 3, dropout=False),
            block(ch, ch, ks, 3),
            block(ch, ch, ks, 3),
            block(ch, ch, ks, 3),
        )

        # compute pattern length after output from convolutional layers
        lin_size = in_size
        for _ in self.conv:
            lin_size = lin_size // 3
        lin_size *= ch

        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(lin_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 3),
        )

        self.temperature_scale = nn.Sequential(
            ScaleConstant(),
        )

    def forward(self, X: Tensor) -> Tensor:
        X = self.conv(X)
        X = self.lin(X)
        X = self.temperature_scale(X)
        return X

    def class_probabilities(self, X: Tensor) -> Tensor:
        return torch.nn.functional.softmax(self.forward(X), dim=1)
