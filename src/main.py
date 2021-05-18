import os
import torch
from torch import nn
from models import JITGNN
from datasets import ASTDataset
from train import train, test, resume_training

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    epochs = 20
    batch_size = 1
    n_classes = 2
    data_dict = {
        'train': '/openstack_train.json',
        'val': '/openstack_valid.json',
        'test': '/openstack_test.json',
        'labels': '/openstack_labels.json'
    }
    dataset = ASTDataset(data_dict)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 1     # plus supernode node feature
    message_size = 32

    model = JITGNN(hidden_size, message_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training
    stats = train(model, optimizer, criterion, epochs, dataset)
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
    # model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))  # need map_location=torch.device('cpu') if on CPU
    # test(model, dataset)
