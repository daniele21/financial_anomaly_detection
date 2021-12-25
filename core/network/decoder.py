# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LSTM_Decoder(nn.Module):

    def __init__(self,
                 in_features,
                 out_feature,
                 seq_len,
                 increase_factor: int,
                 layers: int,
                 seed: int = 2021):

        super(LSTM_Decoder, self).__init__()
        self.name = 'LSTM Decoder'

        self.in_feature = in_features
        self.out_feature = out_feature
        self.increase_factor = increase_factor
        self.layers = layers
        self.seq_len = seq_len

        # self.window = params['window']-1 if params['window'] is not None else None
        torch.manual_seed(seed)

        self.lstm_layers = []
        for layer in range(layers):
            hidden_size = in_features * increase_factor
            self.lstm_layers.append(nn.LSTM(input_size=in_features,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            batch_first=True))
            in_features = hidden_size

        self.output_layer = nn.Linear(hidden_size, out_feature)
        self.final_scale = False if hidden_size == out_feature else True

        for i, lstm_layer in enumerate(self.lstm_layers):
            self.add_module(f'LSTM Layer - {i}', lstm_layer)

    def forward(self, x):
        x = x.repeat(1, self.seq_len, 1)
        # x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        for lstm_layer in self.lstm_layers:
            y_lstm, (h_state, c_state) = lstm_layer(x)

        # x = x.reshape((self.seq_len, self.hidden_dim))

        out = self.output_layer(y_lstm) if self.final_scale else x

        return out
