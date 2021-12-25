# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LSTM_Encoder(nn.Module):

    def __init__(self,
                 in_features,
                 reduce_factor: int,
                 layers: int,
                 seed: int = 2021):

        super(LSTM_Encoder, self).__init__()
        self.name = 'LSTM Encoder'
        self.in_feature = in_features

        # self.window = params['window']-1 if params['window'] is not None else None
        torch.manual_seed(seed)

        self.lstm_layers = []
        for layer in range(layers):
            hidden_size = in_features // reduce_factor
            self.lstm_layers.append(nn.LSTM(input_size=in_features,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            batch_first=True))
            in_features = hidden_size
            self.last_size = hidden_size

        for i, lstm_layer in enumerate(self.lstm_layers):
            self.add_module(f'LSTM Layer - {i}', lstm_layer)

    def forward(self, x):
        h_state = None

        # x = x.unsqueeze(0)
        y_lstm = x
        for lstm_layer in self.lstm_layers:
            y_lstm, (h_state, c_state) = lstm_layer(y_lstm)

        out = h_state

        return out
