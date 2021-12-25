import torch


def data_to_tensor(data):
    return torch.Tensor(data.values)
