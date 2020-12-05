import os

import torch
from datasets import ASTDataset

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


def train(model, optimizer, criterion,
          epochs, n_classes, hidden_size, message_size, n_timesteps, filenames):

    for e in range(epochs):
        print('epoch', e, '\n')
        model = JITGNN(n_classes, hidden_size, message_size, n_timesteps)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters())

        total_loss = 0
        total_files = 0
        model.train()
        for f in filenames:
            dataset = ASTDataset(os.path.join(data_path, f))
            dataset_size = len(dataset)

            for i in range(len(dataset)):
                print('commit', i)
                data = dataset[i]
                if data is None:
                    print()
                    continue
                for chngd_file_tnsr in data:
                    print('src tensor loaded')
                    optimizer.zero_grad()
                    model = model.to(device)
                    outputs = model(chngd_file_tnsr[0].to(device), chngd_file_tnsr[1].to(device),
                                    chngd_file_tnsr[2].to(device), chngd_file_tnsr[3].to(device))
                    loss = criterion(outputs.unsqueeze(0), torch.LongTensor([chngd_file_tnsr[4]]).to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_files += 1
                    print(loss.item())
                    print()

        print('\nepoch avg loss:', str(total_loss / total_files))
        torch.save(model, BASE_PATH + '/trained_models/model.pt')
        print('* model saved.\n')


if __name__ == '__main__':
    print()
