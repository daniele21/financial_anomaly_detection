import unittest

import torch
# from core.network.torch_summary import summary
# from core.network.summary import summary
from pytorch_model_summary import summary

from core.network.autoencoder import LSTM_Autoencoder
from core.network.decoder import LSTM_Decoder
from core.network.encoder import LSTM_Encoder


class TestNetworks(unittest.TestCase):
    window_size = 10
    features = 7
    factor = 2
    layers = 1
    seed = 2021

    def test_lstm_encoder(self):
        encoder = LSTM_Encoder(in_features=self.features,
                               reduce_factor=self.factor,
                               layers=self.layers,
                               seed=self.seed).to(torch.device('cpu'))

        summary(encoder, torch.zeros((1, self.window_size, self.features)),
                show_input=True, print_summary=True)

        summary(encoder, torch.zeros((1, self.window_size, self.features)),
                show_input=False, print_summary=True)

    def test_lstm_decoder(self):
        in_features = (self.features // (self.factor ** self.layers))

        decoder = LSTM_Decoder(in_features=in_features,
                               out_feature=self.features,
                               seq_len=self.window_size,
                               increase_factor=self.factor,
                               layers=self.layers,
                               seed=self.seed).to(torch.device('cpu'))

        summary(decoder, torch.zeros((1, 1, in_features)),
                show_input=True, print_summary=True)

        summary(decoder, torch.zeros((1, 1, in_features)),
                show_input=False, print_summary=True)

    def test_lstm_autoencoder(self):
        device = torch.device('cpu')

        autoencoder = LSTM_Autoencoder(in_features=self.features,
                                       factor=self.factor,
                                       seq_len=self.window_size,
                                       layers=self.layers,
                                       seed=self.seed,
                                       device=device).to(device)

        summary(autoencoder, torch.zeros((1, self.window_size, self.features)),
                show_input=True, print_summary=True)

        summary(autoencoder, torch.zeros((1, self.window_size, self.features)),
                show_input=False, print_summary=True)


if __name__ == '__main__':
    unittest.main()
