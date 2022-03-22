from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn

Model = {
    'GRU': nn.GRU,
    'RNN': nn.RNN,
    'LSTM': nn.LSTM
}

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        model: str,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        bidirect_type: str,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.bidirect_type = bidirect_type
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.input_size = embeddings.shape[1]
        # model architecture
        # self.cnn = nn.Conv1d(
        #     in_channels=self.embed.embedding_dim,
        #     out_channels=self.embed.embedding_dim,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2, # 'same'
        #     padding_mode='zeros',
        # )
        self.rnn = Model[model](
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
        # calculate the output dimension of rnn
        return (int(self.bidirectional)+1) * self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # implement model forward
        # print(batch.shape) # (128, 1X-2X) = (batch_size, max_len)
        x = self.embed(batch)
        # print(x.shape) # (batch_size, max_len, self.embed.embedding_dim=300)
        # x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        r_out, _ = self.rnn(x, None)
        if self.bidirectional:
            if self.bidirect_type == 'mean':
                out = self.out(torch.mean(r_out, dim=1)) # mean
            if self.bidirect_type == 'concate':
                # reference: https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66
                r_out = torch.cat((r_out[:, -1, :self.hidden_size], r_out[:, 0, self.hidden_size:]), dim=1)
                out = self.out(r_out)
        else:
            out = self.out(r_out[:, -1, :])
        return out


class SeqSlotClassifier(torch.nn.Module):
    def __init__(
        self,
        model: str,
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
        # model architecture
        # self.cnn = nn.Conv1d(
        #     in_channels=self.embed.embedding_dim,
        #     out_channels=self.embed.embedding_dim,
        #     kernel_size=5,
        #     stride=1,
        #     padding=2, # 'same'
        #     padding_mode='zeros',
        # )
        self.rnn = Model[model](
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
        # calculate the output dimension of rnn
        return (int(self.bidirectional)+1) * self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # implement model forward
        x = self.embed(batch)
        # x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        r_out, _ = self.rnn(x, None)
        # print(r_out.shape) # (batch_size, max_len, encode_output_size*hidden_layer)
        out = self.out(r_out)
        return out