import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sentence_transformers import SentenceTransformer
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.utils import get_device

from cogrip.language import decode_sent


class LanguageExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, language_dims: int):
        super().__init__(observation_space, language_dims)
        self.device = get_device("auto")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # the policy model calls preprocess_obs which converts Box features to float()
        # we have to undo this operation
        observations = observations.long()
        return self._on_forward(observations)

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SentenceBERT(LanguageExtractor):

    def __init__(self, observation_space: gym.spaces.Box, language_dims: int):
        super().__init__(observation_space, language_dims)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        d_model = self.model.get_sentence_embedding_dimension()
        for param in self.model.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(nn.Linear(d_model, language_dims), nn.ReLU())

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = [decode_sent(o.cpu().numpy()) for o in observations]  # model expects strings
        sentence_embedding = self.model.encode(observations, convert_to_tensor=True)
        sentence_embedding = self.projection(sentence_embedding)
        return sentence_embedding


class SentenceBOW(LanguageExtractor):
    """ Treat a sentence as an unordered bag-of-words (word exists) vector and train a feedforward network on it"""

    def __init__(self, observation_space: gym.spaces.Box, language_dims: int):
        super().__init__(observation_space, language_dims)
        self.vocab_size = observation_space.high[0]
        self.projection = nn.Sequential(nn.Linear(self.vocab_size, language_dims),
                                        nn.ReLU(),
                                        nn.Linear(language_dims, language_dims),
                                        nn.ReLU()
                                        )

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        sentence_embedding = torch.zeros(size=(batch_size, self.vocab_size), device=self.device)

        for idx, o in enumerate(observations):
            """ debug 
            print(o)
            """
            sentence_embedding[idx][o] = 1
        """ debug 
        print()
        for s in sentence_embedding:
            print(s)
        """
        sentence_embedding = self.projection(sentence_embedding)
        return sentence_embedding


class OneHotTextEncoder(LanguageExtractor):
    """ Treat each word as a one-hot vector and train a language model on it"""

    def __init__(self, observation_space: gym.spaces.Box, language_dims: int):
        super().__init__(observation_space, language_dims)
        self.vocab_size = observation_space.high[0]
        self.text_rnn = nn.GRU(self.vocab_size, language_dims, batch_first=True)

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        word_embeddings = F.one_hot(observations, num_classes=self.vocab_size)
        _, hidden = self.text_rnn(word_embeddings.float())
        return hidden[-1]


class TextEncoder(LanguageExtractor):
    """ Treat each word as a word embedding and train a language model on it"""
    """ From rl-starter-files (torch-ac)"""

    def __init__(self, observation_space: gym.spaces.Box, language_dims: int):
        super().__init__(observation_space, language_dims)
        self.vocab_size = observation_space.high[0]
        self.word_embedding = nn.Embedding(self.vocab_size, language_dims, padding_idx=0)
        self.text_rnn = nn.GRU(language_dims, language_dims, batch_first=True)

    def _on_forward(self, observations: torch.Tensor) -> torch.Tensor:
        word_embeddings = self.word_embedding(observations)
        _, hidden = self.text_rnn(word_embeddings)
        return hidden[-1]
