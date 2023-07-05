import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn

from neuact.agents.extractors import initialize_parameters
from neuact.agents.extractors.fusion import FiLM
from neuact.agents.extractors.language import TextEncoder, OneHotTextEncoder, SentenceBOW, SentenceBERT
from neuact.agents.extractors.vision import PixelImageEncoder, SymbolicImageEncoder


class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict,
                 vision_arch: str, language_arch: str,
                 vision_dims: int, language_dims: int,
                 use_feedback: bool, use_mission: bool,
                 fusion_arch: str):
        super().__init__(observation_space, features_dim=1)  # dummy value
        self.vision_arch = vision_arch
        self.language_arch = language_arch
        self.logger = None  # could be set later (from outside)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "vision":
                extractors[key] = self._create_vision_encoder(subspace, vision_dims, fusion_arch)
                total_concat_size += vision_dims
            elif key == "gr_coords":
                extractors[key] = nn.Sequential(nn.Linear(in_features=2, out_features=vision_dims),
                                                nn.LayerNorm(vision_dims))
                total_concat_size += vision_dims
            elif key == "mission":
                if use_mission:
                    print("Using mission...")
                    extractors[key] = self._create_text_encoder(language_dims, subspace)
                    total_concat_size += language_dims
            elif key == "feedback":
                if use_feedback:
                    print("Using feedback...")
                    extractors[key] = self._create_text_encoder(language_dims, subspace)
                    total_concat_size += language_dims
            else:
                raise ValueError("Unknown observation subspace:", key)
        self.extractors = nn.ModuleDict(extractors)

        self.fusion_arch = fusion_arch
        assert fusion_arch in ["linear", "film"]
        if fusion_arch == "linear":
            self.fusion = nn.Sequential(
                nn.Linear(total_concat_size, total_concat_size),
                nn.ReLU()
            )
            # Note: feature_dims is used to init the policy mlp-extractor; duplicates for linear
            self._features_dim = total_concat_size
        if fusion_arch == "film":
            self.endpool = True
            num_module = 1
            self.mission_attention = nn.Sequential(nn.Linear(in_features=vision_dims, out_features=1),
                                                   nn.Sigmoid())
            self.mission_controllers = []
            if use_mission:
                for ni in range(num_module):
                    mod = FiLM(
                        in_features=language_dims,
                        out_features=vision_dims,
                        in_channels=vision_dims, imm_channels=vision_dims)
                    self.mission_controllers.append(mod)
                    self.add_module('FiLM_mission_' + str(ni), mod)
            self.feedback_attention = nn.Sequential(nn.Linear(in_features=vision_dims,
                                                              out_features=int(vision_dims / 2)),
                                                    nn.ReLU(),
                                                    nn.Linear(in_features=int(vision_dims / 2),
                                                              out_features=1),
                                                    nn.Sigmoid())
            self.feedback_controllers = []
            if use_feedback:
                for ni in range(num_module):
                    mod = FiLM(
                        in_features=language_dims,
                        out_features=vision_dims,
                        in_channels=vision_dims, imm_channels=vision_dims)
                    self.feedback_controllers.append(mod)
                    self.add_module('FiLM_feedback_' + str(ni), mod)
            if self.endpool:
                self.film_pool = nn.AdaptiveMaxPool2d((1, 1))
            else:
                self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            self.film_relu = nn.ReLU()
            self.film_norm = nn.LayerNorm(vision_dims)
            # Note: feature_dims is used to init the policy mlp-extractor; stays the same for film
            self._features_dim = vision_dims
        self.apply(initialize_parameters)

    def _create_vision_encoder(self, subspace, vision_dims, fusion_arch):
        valid_values = ["pixels", "symbols"]
        assert self.vision_arch in valid_values, f"language_arch is {self.vision_arch} but must be {valid_values}"
        if self.vision_arch == "pixels":
            print("Using pixel encoder...")
            return PixelImageEncoder(subspace, vision_dims=vision_dims, fusion_arch=fusion_arch)
        if self.vision_arch == "symbols":
            print("Using symbols encoder...")
            return SymbolicImageEncoder(subspace, vision_dims=vision_dims, fusion_arch=fusion_arch)

    def _create_text_encoder(self, language_dims, subspace):
        valid_values = ["we+lm", "oh+lm", "sent-bow", "sent-pre"]
        assert self.language_arch in valid_values, f"language_arch is {self.language_arch} but must be {valid_values}"
        if self.language_arch == "we+lm":
            return TextEncoder(subspace, language_dims=language_dims)
        if self.language_arch == "oh+lm":
            return OneHotTextEncoder(subspace, language_dims=language_dims)
        if self.language_arch == "sent-bow":
            return SentenceBOW(subspace, language_dims=language_dims)
        if self.language_arch == "sent-pre":
            return SentenceBERT(subspace, language_dims=language_dims)

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        key = "vision"
        vision_embeddings = None
        if key in self.extractors:
            vision_embeddings = self.extractors[key](observations[key])
            encoded_tensor_list.append(vision_embeddings)
        assert vision_embeddings is not None

        key = "mission"
        mission_embeddings = None
        if key in self.extractors:
            mission_embeddings = self.extractors[key](observations[key])
            encoded_tensor_list.append(mission_embeddings)

        key = "feedback"
        feedback_embeddings = None
        if key in self.extractors:
            feedback_embeddings = self.extractors[key](observations[key])
            encoded_tensor_list.append(feedback_embeddings)

        key = "gr_coords"
        gr_coords_embeddings = None
        if key in self.extractors:
            gr_coords = observations[key]
            gr_coords_embeddings = self.extractors[key](gr_coords)
            encoded_tensor_list.append(gr_coords_embeddings)

        x = None
        if self.fusion_arch == "linear":
            x = torch.cat(encoded_tensor_list, dim=1)
            x = self.fusion(x)

        if self.fusion_arch == "film":
            if mission_embeddings is not None:
                filmed_vision_embeddings = vision_embeddings
                for controller in self.mission_controllers:
                    mission_out = controller(filmed_vision_embeddings, mission_embeddings)
                    mission_out = mission_out + filmed_vision_embeddings  # residual connection
                    filmed_vision_embeddings = mission_out
                mi = self.film_relu(self.film_pool(filmed_vision_embeddings))
                mi = mi.reshape(mi.shape[0], -1)  # B x 128
                x = mi
            if feedback_embeddings is not None:
                filmed_vision_embeddings = vision_embeddings
                for controller in self.feedback_controllers:
                    feedback_out = controller(filmed_vision_embeddings, feedback_embeddings)
                    feedback_out = feedback_out + filmed_vision_embeddings  # residual connection
                    filmed_vision_embeddings = feedback_out
                fb = self.film_relu(self.film_pool(filmed_vision_embeddings))  # B x 128 x 6 x 6 -> B x 128 x 1 x 1
                fb = fb.reshape(fb.shape[0], -1)  # B x 128
                x = fb
            if mission_embeddings is not None and feedback_embeddings is not None:
                x = (mi + fb) / 2
            x = self.film_norm(x)
        x = (x + gr_coords_embeddings) / 2
        assert x is not None
        return x
