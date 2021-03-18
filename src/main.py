import os
import torch
from torch import nn
from models import JITGNN
from train import train, test, resume_training

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    epochs = 20
    batch_size = 1
    n_classes = 2
    hidden_size = 768
    message_size = 256
    n_timesteps = 4
    train_filename = 'subtrees_0.25_train.json'
    val_filename = 'subtrees_0.25_val.json'
    test_filename = 'subtrees_0.25_test.json'

    model = JITGNN(hidden_size, message_size, n_timesteps)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training
    stats = train(model, optimizer, criterion, epochs, train_filename, val_filename)
    # plot_training(stats)

    # resume training
    # print('resume training')
    # checkpoint = torch.load(os.path.join(BASE_PATH, 'trained_models/checkpoint.pt'))
    # print('checkpoint loaded.')
    # saved_stats = torch.load(os.path.join(BASE_PATH, 'trained_models/stats.pt'))
    # print('stats loaded.')
    # stats = resume_training(checkpoint, saved_stats, model, optimizer, criterion, epochs, train_filename, val_filename)
    # plot_training(stats)

    # testing
    model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))  # need map_location=torch.device('cpu') if on CPU
    test(model, test_filename)
