import gym
import torch
import torchvision.transforms
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torchvision.transforms import InterpolationMode

from neuact.agents.extractors import initialize_parameters


class ImageBOWEmbedding(nn.Module):
    # From BabyAI
    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
        inputs = (inputs + offsets[None, :, None, None]).long()
        inputs = self.embedding(inputs)
        inputs = inputs.sum(1)
        inputs = inputs.permute(0, 3, 1, 2)
        return inputs


class SymbolicImageEncoder(BaseFeaturesExtractor):
    # similar to in BabyAI (added AdaptiveMaxPool2d)

    def __init__(self, observation_space: gym.spaces.Box, vision_dims: int, fusion_arch: str):
        super().__init__(observation_space, vision_dims)
        self.fusion_arch = fusion_arch
        self.cnn = nn.Sequential(
            ImageBOWEmbedding(12, embedding_dim=vision_dims),
            nn.Conv2d(in_channels=vision_dims, out_channels=vision_dims, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(vision_dims),
            nn.ReLU(),
            nn.Conv2d(in_channels=vision_dims, out_channels=vision_dims, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(vision_dims),
            nn.ReLU()
        )
        # for film do not pool yet
        if self.fusion_arch == "linear":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations_long = observations.long()
        # utils.debug_grid(observations_long.squeeze())
        x = self.cnn(observations_long)
        if self.fusion_arch == "linear":
            x = self.pool(x)
            # squeeze but keep batch dimension also with single obs batches
            batch = x.shape[0]
            x = x.view(batch, -1)
        return x


class PixelImageEncoder(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, vision_dims: int, fusion_arch: str, endpool: bool = True):
        super().__init__(observation_space, vision_dims)
        self.fusion_arch = fusion_arch
        if fusion_arch == "film":  # make sure that the output matches the film layer; babyai had a fix 128
            self.cnn = nn.Sequential(*[
                # 56x56 -> 7x7
                # 88x88 -> 11x11
                # adjust kernel and stride to size 4 so that 44x44 -> 11x11
                *([nn.Conv2d(
                    in_channels=3, out_channels=vision_dims, kernel_size=(4, 4),
                    stride=4, padding=0)]),
                nn.Conv2d(in_channels=vision_dims, out_channels=vision_dims,
                          kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
                nn.BatchNorm2d(vision_dims),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
                nn.Conv2d(in_channels=vision_dims, out_channels=vision_dims, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(vision_dims),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
            ])
        else:  # for linear we project later towards vision_dims # NatureCNN
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU())
        # babyAI uses 56 b.c. cnn maps to 7x7 (field-of-view size)
        # O = (W - K) / S + 1
        # ((O - 1) * S) + K = W
        # ((11 -1) * 8) + 8 = 88
        # we use 88 which maps to 11x11 (field-of-view size)
        # assume smaller kernel 4
        # ((11 -1) * 4) + 4 = 44
        self.size = 44

        self.resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST),
        ])
        if self.fusion_arch == "linear":
            self.flatten = nn.Flatten()
            # Compute shape by doing one forward pass
            with torch.no_grad():
                sample = observation_space.sample()
                sample = torch.as_tensor(sample[None]).float()
                sample = self.resize(sample)
                n_flatten = self.flatten(self.cnn(sample)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n_flatten, vision_dims), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.resize(observations)
        """ debugging 
        for o in observations:
            o = o.permute((1, 2, 0))
            plt.imshow(o.numpy())
            plt.show()
        """
        x = self.cnn(observations)
        if self.fusion_arch == "linear":
            x = self.flatten(x)
            x = self.linear(x)
        # for Film with pixels the output should be 128x7x7
        return x
