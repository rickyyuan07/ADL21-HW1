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
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
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
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=self.encoder_output_size, out_features=hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            # nn.LayerNorm(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size//2, out_features=num_class),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return (int(self.bidirectional)+1) * self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # print(batch.shape) # (128, 1X-2X) = (batch_size, max_len)
        x = self.embed(batch)
        r_out, _ = self.rnn(x, None)
        if self.bidirectional:
            # out = self.out(torch.mean(r_out, dim=1)) # mean
            # reference: https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
            r_out = torch.cat((r_out[:, -1, :self.hidden_size], r_out[:, 0, self.hidden_size:]), dim=1)
            out = self.out(r_out)
        else:
            out = self.out(r_out[:, -1, :])
        return out


class SeqSlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqSlotClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
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
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(self.encoder_output_size),
            nn.Linear(in_features=self.encoder_output_size, out_features=num_class),
            # nn.BatchNorm1d(hidden_size//2), 
            # nn.Dropout(dropout),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden_size//2, out_features=num_class),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return (int(self.bidirectional)+1) * self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        r_out, _ = self.rnn(x, None)
        # print(r_out.shape) # (batch_size, max_len, encode_output_size*hidden_layer)
        out = self.out(r_out)
        return out