import math
import os
import time
import numpy as np
import torch
from datasets import ASTDataset
from metrics import roc_auc
# import matplotlib.pyplot as plt


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


def train(model, optimizer, criterion, epochs, train_filename, val_filename, so_far=0, resume=None):

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

    train_dataset = ASTDataset(os.path.join(data_path, train_filename))
    val_dataset = ASTDataset(os.path.join(data_path, val_filename))
    display_every = len(train_dataset) // 100

    print('training')
    for e in range(epochs):
        print('\nepoch {:3d}/{}\n'.format((e + 1 + so_far), (epochs + so_far)))
        # training
        start = time.time()
        total_loss = 0
        n_files = 0
        y_scores = []
        y_true = []

        model.train()
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            if data is None:
                continue
            for file_tensors in data:
                optimizer.zero_grad()
                model = model.to(device)
                output = model(file_tensors[0].to(device), file_tensors[1].to(device),
                               file_tensors[2].to(device), file_tensors[3].to(device))
                loss = criterion(output, torch.Tensor([file_tensors[4]]).to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                y_scores.append(output.item())
                y_true.append(file_tensors[4])
                n_files += 1

                if n_files % display_every == display_every - 1:
                    print('\t[{:5d}/{}]\tloss: {:.4f}'.format(
                        n_files, len(train_dataset), loss.item()))

        print('\nepoch duration: {}'.format(time_since(start)))

        training_loss = total_loss / n_files
        _, _, _, training_auc = evaluate(y_true, y_scores)
        print('\n<==== training loss = {:.4f} ====>'.format(training_loss))
        print('metrics: AUC={}\n'.format(training_auc))

        all_training_losses.append(training_loss)
        all_training_aucs.append(training_auc)

        # validation
        total_loss = 0
        n_files = 0
        y_scores = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for i in range(len(val_dataset)):
                data = val_dataset[i]
                if data is None:
                    continue
                for file_tensors in data:
                    model = model.to(device)
                    output = model(file_tensors[0].to(device), file_tensors[1].to(device),
                                   file_tensors[2].to(device), file_tensors[3].to(device))
                    loss = criterion(output, torch.Tensor([file_tensors[4]]).to(device))
                    total_loss += loss.item()

                    y_scores.append(output.item())
                    y_true.append(file_tensors[4])
                    n_files += 1

        val_loss = total_loss / n_files
        _, _, _, val_auc = evaluate(y_true, y_scores)
        print('\n<==== validation loss = {:.4f} ====>'.format(val_loss))
        print('metrics: AUC={}\n'.format(val_auc))

        if len(all_val_aucs) == 0 or val_auc > max(all_val_aucs):
            torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_best_auc.pt'))
            print('* model_best_auc saved.')
        if len(all_val_losses) == 0 or val_loss < min(all_val_losses):
            torch.save(model, os.path.join(BASE_PATH, 'trained_models/model_least_loss.pt'))
            print('* model_least_loss saved.')

        all_val_losses.append(val_loss)
        all_val_aucs.append(val_auc)

        torch.save({
            'epoch': e + 1 + so_far,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'all_training_losses': all_training_losses,
            'all_training_aucs': all_training_aucs,
            'all_val_losses': all_val_losses,
            'all_val_aucs': all_val_aucs,
        }, 'checkpoint.pt')
        print('* checkpoint saved.\n')

    print('\ntraining finished')
    return all_training_aucs, all_training_losses, all_val_aucs, all_val_losses


def test(model, test_filename):
    test_dataset = ASTDataset(os.path.join(data_path, test_filename))

    print('testing')
    n_files = 0
    y_scores = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            data = test_dataset[i]
            if data is None:
                continue
            for file_tensors in data:
                model = model.to(device)
                output = model(file_tensors[0].to(device), file_tensors[1].to(device),
                               file_tensors[2].to(device), file_tensors[3].to(device))

                y_scores.append(output.item())
                y_true.append(file_tensors[4])
                n_files += 1

    _, _, _, auc = evaluate(y_true, y_scores)
    print('metrics: AUC={}\n'.format(auc))

    print('testing finished')


def resume_training(checkpoint, model, optimizer, criterion, epochs, train_filename, val_filename):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    so_far = checkpoint['epoch']
    resume = {
        'all_training_losses': checkpoint['all_training_losses'],
        'all_training_accs': checkpoint['all_training_accs'],
        'all_val_losses': checkpoint['all_val_losses'],
        'all_val_accs': checkpoint['all_val_accs']
    }
    stats = train(model, optimizer, criterion, epochs, train_filename, val_filename, so_far, resume)
    return stats


# def plot_training(stats):
#     all_training_aucs, all_training_losses, all_val_aucs, all_val_losses = stats
#
#     plt.figure()
#     plt.plot(all_training_losses)
#     plt.plot(all_val_losses)
#     plt.title('Loss')
#     plt.ylabel('Binary Cross Entropy')
#     plt.xlabel('Epochs')
#     plt.legend(['training loss', 'validation loss'], loc='upper right')
#
#     plt.figure()
#     plt.plot(all_training_aucs)
#     plt.plot(all_val_aucs)
#     plt.title('Performance')
#     plt.ylabel('AUC')
#     plt.xlabel('Epochs')
#     plt.legend(['training auc', 'validation auc'], loc='lower right')


if __name__ == '__main__':
    print()
