from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.input_size = embeddings.shape[1]
        # TODO: model architecture
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True, # (batch_size, time_step, input)
            bidirectional=bidirectional,
        )
        self.out = nn.Sequential(nn.Dropout(dropout),
                      nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 2),
                      nn.PReLU(),
                      nn.Dropout(dropout),
                      nn.Linear(in_features=hidden_size * 2, out_features=num_class))

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # print(batch.shape) # (128, 75) = (batch_size, max_len)
        x = self.embed(batch)
        
        r_out, _ = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out
