import math

from sklearn.linear_model import LogisticRegression
import pickle
import torch
import torch.nn as nn
from torch.nn import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        for i in range(self.n_timesteps):
            # take care of the shape in the matrix multiplication.
            current_messages = self.linear(x)
            next_messages = torch.matmul(adj_matrix, current_messages)
            x = self.gru_cell(next_messages, x)

        return x


class GraphConvolution(nn.Module):
    """
    from https://github.com/tkipf/pygcn/
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class JITGNN(nn.Module):
    def __init__(self, hidden_size, message_size, n_timesteps):
        super(JITGNN, self).__init__()
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.n_timesteps = n_timesteps
        self.gnn1 = self.make_gcn()
        self.gnn2 = self.make_gcn()
        self.fc1 = nn.Linear(2 * message_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def make_gcn(self):
        modules = [
            GraphConvolution(self.hidden_size, self.message_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        ]
        for i in range(self.n_timesteps - 1):
            modules += [
                GraphConvolution(self.message_size, self.message_size),
                nn.ReLU()
            ]
            if i < self.n_timesteps - 2:
                modules += [nn.Dropout(0.2)]
        return nn.Sequential(*modules)

    @staticmethod
    def add_supernode(feature, adj_matrix):
        with torch.no_grad:
            zeros = torch.zeros(1, feature.size(1)).to(device)
            feature = torch.cat((feature, zeros), 0)

            zeros = torch.zeros(adj_matrix.size(0), 1).to(device)
            adj_matrix = torch.cat((adj_matrix, zeros), 1)
            ones = torch.ones(1, adj_matrix.size(1)).to(device)
            adj_matrix = torch.cat((adj_matrix, ones), 0)

        return feature, adj_matrix

    @staticmethod
    def normalize(matrix):
        with torch.no_grad():
            row_sum = torch.FloatTensor(matrix.sum(1))
            rs_inverse = torch.pow(row_sum, -1).flatten()
            rs_inverse[torch.isinf(rs_inverse)] = 0.
            rs_mat_inv = torch.diag(rs_inverse)
            matrix = rs_mat_inv.mm(matrix)

        return matrix

    def forward(self, b_x, b_adj, a_x, a_adj):
        print('1.', b_x.requires_grad, b_adj.requires_grad, a_x.requires_grad, a_adj.requires_grad)

        bx, b_adj = self.add_supernode(b_x, b_adj)
        b_adj = self.normalize(b_adj)
        ax, a_adj = self.add_supernode(a_x, a_adj)
        a_adj = self.normalize(a_adj)

        print('2.', b_x.requires_grad, b_adj.requires_grad, a_x.requires_grad, a_adj.requires_grad)

        b_node_embeddings = self.gnn1(b_x, b_adj)
        b_supernode = b_node_embeddings[-1, :]

        print('3.', b_node_embeddings.requires_grad)
        print('4.', b_supernode.requires_grad)

        a_node_embeddings = self.gnn2(a_x, a_adj)
        a_supernode = a_node_embeddings[-1, :]

        print('5.', a_node_embeddings.requires_grad)
        print('6.', a_supernode.requires_grad)

        supernodes = torch.cat((b_supernode, a_supernode), 0)   # maybe a distance measure later

        print('7.', supernodes.requires_grad)

        hidden = self.fc1(supernodes)
        print('8.', hidden.requires_grad)
        hidden = self.relu(hidden)
        print('9.', hidden.requires_grad)
        hidden = self.dropout(hidden)
        print('10.', hidden.requires_grad)
        output = self.fc2(hidden)
        print('10.', output.requires_grad)
        output = self.sigmoid(output)
        print('10.', output.requires_grad)
        return output

