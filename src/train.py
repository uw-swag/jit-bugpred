import math
import os
import time
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from scipy.optimize import differential_evolution
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import pandas as pd
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
    return '{}h {}min {:.2f} sec'.format(h, m, s)


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
        # features_list = []
        # label_list = []

        model.train()
        dataset.set_mode('train')
        print('len(data) is {}'.format(str(len(dataset))))
        for i in range(len(dataset)):
            data = dataset[i]
            label = data[4]
            optimizer.zero_grad()
            model = model.to(device)
            output, features = model(data[0].to(device), data[1].to(device),
                                     data[2].to(device), data[3].to(device))
            # features_list.append(features)
            # label_list.append(label)
            loss = criterion(output, torch.Tensor([label]).to(device))
            loss.backward()
            optimizer.step()

            y_scores.append(torch.sigmoid(output).item())
            y_true.append(label)

            total_loss += loss.item()
            if i % 100:
                print('\t[{:5d}/{}]\tloss: {:.4f}'.format(
                    i, len(dataset), loss.item()))

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
        print('len(data) is {}'.format(str(len(dataset))))
        with torch.no_grad():
            for i in range(len(dataset)):
                data = dataset[i]
                label = data[4]
                model = model.to(device)
                output, features = model(data[0].to(device), data[1].to(device),
                                         data[2].to(device), data[3].to(device))
                # features_list.append(features)
                # label_list.append(label)
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
            # torch.save(torch.vstack(features_list), os.path.join(BASE_PATH, 'trained_models/train_features.pt'))
            # torch.save(torch.Tensor(label_list), os.path.join(BASE_PATH, 'trained_models/train_labels.pt'))
            # print('* features saved.')
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


def objective_func(k, train_features, train_labels, valid_features, valid_labels):
    smote = SMOTE(random_state=42, k_neighbors=int(np.round(k)), n_jobs=32)
    train_feature_res, train_label_res = smote.fit_resample(train_features, train_labels)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(train_feature_res, train_label_res)
    prob = clf.predict_proba(valid_features)[:, 1]
    auc = roc_auc_score(valid_labels, prob)

    return -auc


def train(clf, train_features, train_labels):
    percent_80 = int(train_features.shape[0] * 0.8)
    train_features, valid_features = train_features[:percent_80], train_features[percent_80:]
    train_labels, valid_labels = train_labels[:percent_80], train_labels[percent_80:]
    bounds = [(1, 20)]
    opt = differential_evolution(objective_func, bounds, args=(train_features, train_labels,
                                                               valid_features, valid_labels),
                                 popsize=10, mutation=0.7, recombination=0.3, seed=0)
    smote = SMOTE(random_state=42, n_jobs=32, k_neighbors=int(np.round(opt.x)))
    train_features, train_labels = smote.fit_resample(train_features, train_labels)
    clf.fit(train_features, train_labels)
    prob = clf.predict_proba(valid_features)[:, 1]
    _, _, _, auc = evaluate(valid_labels, prob)
    print('metrics: AUC={}\n'.format(auc))


def test(model, dataset, clf):
    print('testing')
    y_scores = []
    y_true = []
    # features_list = []
    # label_list = []

    model.eval()
    dataset.set_mode('test')
    print('len(data) is {}'.format(str(len(dataset))))
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            label = data[4]
            model = model.to(device)
            output, features = model(data[0].to(device), data[1].to(device),
                                     data[2].to(device), data[3].to(device))
            # features_list.append(features)
            # label_list.append(label)
            y_scores.append(torch.sigmoid(output).item())
            y_true.append(label)

    pd.DataFrame({'y_true': y_true, 'y_score': y_scores}).to_csv(os.path.join(data_path, 'test_result.csv'))
    fpr, tpr, thresholds, auc = evaluate(y_true, y_scores)
    print('metrics: AUC={}\n\nthresholds={}\n'.format(auc, str(thresholds)))
    # features = torch.vstack(features_list).cpu().detach().numpy()
    # labels = torch.Tensor(label_list).cpu().detach().numpy()
    # fpr, tpr, thresholds, auc = evaluate(labels, clf.predict_proba(features)[:, 1])
    # print('metrics: AUC={}\n\nthresholds={}\n'.format(auc, str(thresholds)))

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

    p, r, _ = precision_recall_curve(y_true, y_scores)
    plt.clf()
    plt.title('Precision-Recall')
    plt.plot(r, p, 'b', label='AUC = %0.2f' % metrics.auc(r, p))
    plt.legend(loc='upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(os.path.join(BASE_PATH, 'trained_models/pr.png'))

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
