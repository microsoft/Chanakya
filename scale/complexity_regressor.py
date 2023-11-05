import torch
from torch import mul, nn, numel
import torch.nn.functional as F

import time


class ComplexityRegressor(nn.Module):
    def __init__(
        self,
        in_channels,
        intermed_channels=256,
        out_vals=1,
        out_type="none",
        filters=[1, 3],
        num_levels=1,
    ):
        super().__init__()
        self.out_type = out_type
        self.filters = []
        if 1 in filters:
            self.filters.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, intermed_channels, (1, 1), padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),  # 512, 1, 1
                )
            )
        if 3 in filters:
            self.filters.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, intermed_channels, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),  # 512, 1, 1
                )
            )
        if 5 in filters:
            self.filters.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, intermed_channels, (5, 5), padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),  # 512, 1, 1
                )
            )
        self.filters = nn.Sequential(*self.filters)
        self.fc = nn.Linear(
            len(filters) * num_levels * intermed_channels, out_vals
        )  # N, F*K*C, 1, 1
        self.drop= nn.Dropout(p=0.2)
        self.relu= nn.ReLU()

    def forward_one_level(self, x):
        x = [f(x) for f in self.filters]
        x = torch.cat(x, dim=1)  # N, F*C
        return x

    def forward(self, x):
        y = []
        if len(x[0].size()) == 5:
            for x_lev in x:           
                y.append(torch.squeeze(x_lev))
            x = y
            y = []
        if len(x[0].size())==3:
            for x_lev in x:
                y.append(x_lev.unsqueeze(0))
            x = y
            y = []
        x = [self.forward_one_level(x_lev) for x_lev in x]
        x = torch.cat(x, dim=1)  # N, F*C*K
        
        x = x.reshape(x.size(0), -1)  # N, F*C*K, 1, 1 -> N, F*C*K
        x = self.fc(x)
        if self.out_type == "none":
            return x
        elif self.out_type == "tanh":
            return torch.tanh(x)
        elif self.out_type == "sigmoid":
            return torch.sigmoid(x)
        elif self.out_type == "relu":
            return F.relu(x)


def build_complexity_regressor(regressor_info):
    return ComplexityRegressor(
        regressor_info["input_channels"],
        regressor_info["intermed_channels"],
        regressor_info["num_outputs"],
        regressor_info["outputs_activation"],
        regressor_info["filters"],
        regressor_info["num_levels"],
    )

if __name__ == "__main__":
    device = torch.device("cuda:2")
    feats = [
        torch.rand(1, 256, 48, 60).to(device),
        torch.rand(1, 256, 24, 30).to(device),
        torch.rand(1, 256, 12, 15).to(device),
        torch.rand(1, 256, 6, 8).to(device),
        torch.rand(1, 256, 3, 4).to(device),
    ]
    regressor = ComplexityRegressor(256, 256, 1, "tanh", [1, 3], 5)
    regressor.to(device)
    print(regressor)
    for i in range(10):
        t1 = time.time()
        y = regressor(feats)
        print((time.time() - t1) * 1000)
        print(y)
