from torch import nn

from core.network.decoder import LSTM_Decoder
from core.network.encoder import LSTM_Encoder


class LSTM_Autoencoder(nn.Module):

    def __init__(self, in_features, seq_len, factor, layers, seed, device):
        super(LSTM_Autoencoder, self).__init__()

        self.encoder = LSTM_Encoder(in_features=in_features,
                                    reduce_factor=factor,
                                    layers=layers,
                                    seed=seed).to(device)

        decoder_features = (in_features // (factor ** layers))
        self.decoder = LSTM_Decoder(in_features=decoder_features,
                                    out_feature=in_features,
                                    seq_len=seq_len,
                                    increase_factor=factor,
                                    layers=layers,
                                    seed=seed).to(device)

    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)

        return z, x_prime
