import math
import os
import time
import numpy as np
import torch

from metrics import roc_auc
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


def time_since(since):
    now = time.time()
    s = now - since
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '{}h {}min {}sec'.format(h, m, s)


def evaluate(label, output):
    return roc_auc(np.array(label), np.array(output))


def pretrain(model, optimizer, criterion, epochs, dataset, so_far=0, resume=None):
    if resume:
        all_training_aucs = resume['all_training_aucs']
        all_training_losses = resume['all_training_losses']
        all_val_aucs = resume['all_val_aucs']
        all_val_losses = resume['all_val_losses']
    else:
        all_training_aucs = []
        all_training_losses = []
        all_val_aucs = []
        all_val_losses = []

    # display_every = len(train_dataset) // 100

    print('training')
    for e in range(epochs):
        print('\nepoch {:3d}/{}\n'.format((e + 1 + so_far), (epochs + so_far)))
        # training
        start = time.time()
        total_loss = 0
        y_scores = []
        y_true = []
        features_list = []
        label_list = []

        model.train()
        dataset.set_mode('train')
        for i in range(len(dataset)):
            data = dataset[i]
            label = data[4]
            commit_loss = 0
            optimizer.zero_grad()
            model = model.to(device)
            output, features = model(data[0].to(device), data[1].to(device),
                                     data[2].to(device), data[3].to(device))
            features_list.append(features)
            label_list.append(label)
            loss = criterion(output, torch.Tensor([label]).to(device))
            loss.backward()
            optimizer.step()
            commit_loss += loss.item()

            y_scores.append(torch.sigmoid(output).item())
            y_true.append(label)

            mean_commit_loss = commit_loss / len(data)
            total_loss += mean_commit_loss
            if label:
                print('\t[{:5d}/{}]\tloss: {:.4f}'.format(
                    i, len(dataset), mean_commit_loss))

        print('\nepoch duration: {}'.format(time_since(start)))

        torch.save({
            'epoch': e + 1 + so_far,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(BASE_PATH, 'trained_models/checkpoint.pt'))
        print('* checkpoint saved.')

        training_loss = total_loss / len(dataset)
        _, _, _, training_auc = evaluate(y_true, y_scores)
        print('\n<==== training loss = {:.4f} ====>'.format(training_loss))
        print('metrics: AUC={}'.format(training_auc))

        all_training_losses.append(training_loss)
        all_training_aucs.append(training_auc)

        # validation
        total_loss = 0
        y_scores = []
        y_true = []

        model.eval()
        dataset.set_mode('val')
        with torch.no_grad():
            for i in range(len(dataset)):
                data = dataset[i]
                label = data[4]
                model = model.to(device)
                output, features = model(data[0].to(device), data[1].to(device),
                                         data[2].to(device), data[3].to(device))
                features_list.append(features)
                label_list.append(label)
                loss = criterion(output, torch.Tensor([label]).to(device))
                total_loss += loss.item()

                y_scores.append(torch.sigmoid(output).item())
                y_true.append(label)

        val_loss = total_loss / len(dataset)
        _, _, _, val_auc = evaluate(y_true, y_scores)
        print('<==== validation loss = {:.4f} ====>'.format(val_loss))
        print('metrics: AUC={}\n'.format(val_auc))

        if len(all_val_aucs) == 0 or val_auc > max(all_val_aucs):
            torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
            print('* model_best_auc saved.')
            torch.save(torch.vstack(features_list), os.path.join(BASE_PATH, 'trained_models/train_features.pt'))
            torch.save(torch.Tensor(label_list), os.path.join(BASE_PATH, 'trained_models/train_labels.pt'))
            print('* features saved.')
        if len(all_val_losses) == 0 or val_loss < min(all_val_losses):
            torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_least_loss.pt'))
            print('* model_least_loss saved.')

        all_val_losses.append(val_loss)
        all_val_aucs.append(val_auc)

        torch.save({
            'all_training_losses': all_training_losses,
            'all_training_aucs': all_training_aucs,
            'all_val_losses': all_val_losses,
            'all_val_aucs': all_val_aucs,
        }, os.path.join(BASE_PATH, 'trained_models/stats.pt'))
        print('* stats saved.\n')

    torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_final.pt'))
    print('* model_final saved.')
    print('\ntraining finished')


def train(clf, train_features, train_labels):
    percent_80 = int(train_features.shape[0] * 0.8)
    clf.fit(train_features[:percent_80], train_labels[:percent_80])
    prob = clf.predict_proba(train_features[percent_80:])[:, 1]
    _, _, _, auc = evaluate(train_labels[percent_80:], prob)
    print('metrics: AUC={}\n'.format(auc))


def test(model, dataset, clf):
    print('testing')
    features_list = []
    label_list = []

    model.eval()
    dataset.set_mode('test')
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            label = data[4]
            model = model.to(device)
            _, features = model(data[0].to(device), data[1].to(device),
                                data[2].to(device), data[3].to(device))
            features_list.append(features)
            label_list.append(label)

    features = torch.vstack(features_list).cpu().detach().numpy()
    labels = torch.Tensor(label_list).cpu().detach().numpy()
    fpr, tpr, thresholds, auc = evaluate(labels, clf.predict_proba(features)[:, 1])
    print('metrics: AUC={}\n\nthresholds={}\n'.format(auc, str(thresholds)))

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/roc.png'))

    print('testing finished')


def resume_training(checkpoint, stats, model, optimizer, criterion, epochs, dataset):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    so_far = checkpoint['epoch']
    resume = {
        'all_training_losses': stats['all_training_losses'],
        'all_training_aucs': stats['all_training_aucs'],
        'all_val_losses': stats['all_val_losses'],
        'all_val_aucs': stats['all_val_aucs']
    }
    print('all set ...')
    pretrain(model, optimizer, criterion, epochs, dataset, so_far, resume)


def plot_training(stats):
    all_training_aucs = stats['all_training_aucs']
    all_training_losses = stats['all_training_losses']
    all_val_aucs = stats['all_val_aucs']
    all_val_losses = stats['all_val_losses']

    plt.figure()
    plt.plot(all_training_losses)
    plt.plot(all_val_losses)
    plt.title('Loss')
    plt.ylabel('Binary Cross Entropy')
    plt.xlabel('Epochs')
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/loss.png'))

    plt.figure()
    plt.plot(all_training_aucs)
    plt.plot(all_val_aucs)
    plt.title('Performance')
    plt.ylabel('AUC')
    plt.xlabel('Epochs')
    plt.legend(['training auc', 'validation auc'], loc='lower right')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/performance.png'))


if __name__ == '__main__':
    print()
