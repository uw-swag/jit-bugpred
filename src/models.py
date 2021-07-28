import math

from sklearn.linear_model import LogisticRegression
import pickle
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

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

    def forward(self, feature, adj):
        support = torch.mm(feature, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output += self.bias
        output = F.relu(output)
        output = F.dropout(output, p=0.2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class JITGNN(nn.Module):
    def __init__(self, hidden_size, message_size):
        super(JITGNN, self).__init__()
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.gnn11 = GraphConvolution(hidden_size, message_size)
        self.gnn12 = GraphConvolution(message_size, message_size)
        self.gnn13 = GraphConvolution(message_size, message_size)
        self.gnn14 = GraphConvolution(message_size, message_size)
        self.gnn21 = GraphConvolution(hidden_size, message_size)
        self.gnn22 = GraphConvolution(message_size, message_size)
        self.gnn23 = GraphConvolution(message_size, message_size)
        self.gnn24 = GraphConvolution(message_size, message_size)
        self.fc = nn.Linear(2 * message_size + 22, 1)

    def forward(self, b_x, b_adj, a_x, a_adj):
        # change the design here. add adjacency matrix to graph convolution class so not pass it every time.
        b_node_embeddings = self.gnn14(self.gnn13(self.gnn12(self.gnn11(b_x, b_adj), b_adj), b_adj), b_adj)
        b_supernode = b_node_embeddings[-1, :]
        a_node_embeddings = self.gnn24(self.gnn23(self.gnn22(self.gnn21(a_x, a_adj), a_adj), a_adj), a_adj)
        a_supernode = a_node_embeddings[-1, :]
        supernodes = torch.hstack([b_supernode, a_supernode])   # maybe a distance measure later

        output = self.fc(supernodes)
        return output, supernodes

