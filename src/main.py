import torch
from torch import nn

from models import JITGNN
from train import train

if __name__ == '__main__':
    epochs = 5
    batch_size = 1
    n_classes = 2
    hidden_size = 768
    message_size = 256
    n_timesteps = 4
    filenames = ['asts_200.json']

    model = JITGNN(n_classes, hidden_size, message_size, n_timesteps)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(model, optimizer, criterion, epochs, n_classes, hidden_size, message_size, n_timesteps, filenames)
