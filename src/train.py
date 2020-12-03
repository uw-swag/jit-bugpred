print('here we are')
import os

import torch
from torch import nn
from datasets import ASTDataset
from models import JITGNN

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


if __name__ == '__main__':
    epochs = 5
    batch_size = 1
    n_classes = 2
    hidden_size = 768
    message_size = 768
    n_timesteps = 4

    filenames = ['asts_200.json']

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

        print('\nepoch avg loss:', str(total_loss / total_files), '\n')
        torch.save(model, data_path + '/model.pt')
