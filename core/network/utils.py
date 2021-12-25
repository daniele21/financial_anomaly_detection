from torch import nn


def init_weights(module):
    if isinstance(module, nn.LSTM):
        module.reset_parameters()
        # nn.init.normal_(module.weight, mean=0.0, std=0.1)  ## or simply use your layer.reset_parameters()
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(1 / module.in_features))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.normal_(module.weight, mean=0.0, std=np.sqrt(4 / module.in_channels))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
