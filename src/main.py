import os
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from models import JITGNN
from datasets import ASTDataset
from train import pretrain, test, resume_training, plot_training, train
import argparse

BASE_PATH = os.path.dirname(os.path.dirname(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    epochs = 30
    batch_size = 1
    n_classes = 2
    data_dict = {
        'train': ['/balance_train_1.json', '/balance_train_2.json', '/balance_train_3.json'],
        'val': ['/balance_valid.json'],
        'test': ['/openstack_test_color.json'],
        'labels': ['/balance_labels.json', '/openstack_labels.json']
    }
    commit_lists = {
        'train': '/balance_train.csv',
        'val': '/balance_valid.csv',
        'test': '/openstack_test.csv'
    }
    dataset = ASTDataset(data_dict, commit_lists, special_token=True, cross_lingual=True)
    hidden_size = len(dataset.vectorizer_model.vocabulary_) + 2    # plus supernode node feature and node colors
    print('hidden_size is {}'.format(hidden_size))
    message_size = 32

    model = JITGNN(hidden_size, message_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training
    pretrain(model, optimizer, criterion, epochs, dataset)
    # train_features = torch.load(os.path.join(BASE_PATH, 'trained_models/train_features.pt')).cpu().detach().numpy()
    # train_labels = torch.load(os.path.join(BASE_PATH, 'trained_models/train_labels.pt')).cpu().detach().numpy()
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    # train(clf, train_features, train_labels)

    # resume training
    # print('resume training')
    # checkpoint = torch.load(os.path.join(BASE_PATH, 'trained_models/checkpoint.pt'))
    # print('checkpoint loaded.')
    # saved_stats = torch.load(os.path.join(BASE_PATH, 'trained_models/stats.pt'))
    # print('stats loaded.')
    # resume_training(checkpoint, saved_stats, model, optimizer, criterion, epochs, dataset)

    # plotting performance and loss plots
    saved_stats = torch.load(os.path.join(BASE_PATH, 'trained_models/stats.pt'))
    print('stats loaded.')
    plot_training(saved_stats)

    if args.test:
        # need map_location=torch.device('cpu') if on CPU
        model = torch.load(os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
        test(model, dataset, clf)
        test(model, dataset, clf)
        test(model, dataset, clf)

