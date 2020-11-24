from sklearn.linear_model import LogisticRegression
import pickle
import torch
import torch.nn as nn


class LogisticRegressionModel:
    def __init__(self, train_inputs, train_labels, test_inputs, test_labels, save_dir, file_name):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.save_dir = save_dir
        self.file_name = file_name
        self.model = LogisticRegression(random_state=0)

    def train(self):
        self.model.fit(self.train_inputs, self.train_labels)
        with open(self.save_dir + self.file_name, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self):
        with open(self.save_dir + self.file_name, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model.predict(self.test_inputs)


class GatedGNN(nn.Module):
    def __init__(self, hidden_size, message_size, n_timesteps):
        super(GatedGNN, self).__init__()
        # need a matrix multiplication but it's not a layer since we don't need it's gradient
        # I may need detach() or requires_grad = False later if I see this operation has gradient).
        # update: NO! according to what I visualized in tse.ipynb removing matmul from the gradient graph will lead to
        # linear layer removal so definitely no. Besides, there is no reason to remove matmul since it doesn't have
        # any parameters it only does an operation.
        self.linear = nn.Linear(hidden_size, message_size)
        self.gru_cell = nn.GRUCell(message_size, hidden_size)
        self.n_timesteps = n_timesteps

    def forward(self, x, adj_matrix):
        # do we need non-linearity?
        # do we need dropout?
        # do we need batch normalization?
        h = x
        h = torch.cat((h, torch.zeros(1, h.size(1))), 0)
        ones = torch.ones(1, adj_matrix.size(0))
        adj_matrix = torch.cat((adj_matrix, ones), 0)
        zeros = torch.zeros(adj_matrix.size(0), 1)
        adj_matrix = torch.cat((adj_matrix, zeros), 1)
        for i in range(self.n_timesteps):
            # take care of the shape in the matrix multiplication.
            current_messages = self.linear(h)
            next_messages = torch.matmul(adj_matrix, current_messages)
            h = self.gru_cell(next_messages, h)

        return h


class JITGNN(nn.Module):
    def __init__(self, n_classes, hidden_size, message_size, n_timesteps):
        super(JITGNN, self).__init__()
        self.ggnn = GatedGNN(hidden_size, message_size, n_timesteps)
        self.fc = nn.Linear(2 * hidden_size, n_classes)
        self.softmax = nn.LogSoftmax()  # be careful about the loss function. It's LogSoftmax. Loss should be NLLLoss

    def forward(self, b_x, b_adj, a_x, a_adj):
        # consider attention. maybe instead of supernode
        b_node_embeddings = self.ggnn(b_x, b_adj)
        b_supernode = b_node_embeddings[-1, :]
        a_node_embeddings = self.ggnn(a_x, a_adj)
        a_supernode = a_node_embeddings[-1, :]
        supernodes = torch.cat((b_supernode, a_supernode), 0)   # maybe a distance measure later
        return self.softmax(self.fc(supernodes))

