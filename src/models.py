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


class AttentionModule(torch.nn.Module):
    """
    Attention Module to make a pass on graph. from FuncGNN implementation at https://github.com/aravi11/funcGNN/
    """
    def __init__(self, size):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.size = size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.size,
                                                             self.size))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TensorNetworkModule(torch.nn.Module):
    """
    funcGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, hidden_size, neuron_size):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.size = hidden_size
        self.tensor_neurons = neuron_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.size,
                                                             self.size,
                                                             self.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.tensor_neurons,
                                                                   2*self.size))
        self.bias = torch.nn.Parameter(torch.Tensor(self.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(embedding_1.unsqueeze(0), self.weight_matrix.view(self.size, -1))
        scoring = scoring.view(self.size, self.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), torch.t(embedding_2.unsqueeze(0)))
        combined_representation = torch.cat((torch.t(embedding_1.unsqueeze(0)), torch.t(embedding_2.unsqueeze(0))))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        scores = torch.t(scores)
        return scores


class JITGNN(nn.Module):
    def __init__(self, hidden_size, message_size, metric_size):
        super(JITGNN, self).__init__()
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.neuron_size = 32
        self.gnn11 = GraphConvolution(hidden_size, message_size)
        self.gnn12 = GraphConvolution(message_size, message_size)
        self.gnn13 = GraphConvolution(message_size, message_size)
        self.gnn14 = GraphConvolution(message_size, message_size)
        self.gnn21 = GraphConvolution(hidden_size, message_size)
        self.gnn22 = GraphConvolution(message_size, message_size)
        self.gnn23 = GraphConvolution(message_size, message_size)
        self.gnn24 = GraphConvolution(message_size, message_size)
        self.attention = AttentionModule(message_size)
        self.tensor_net = TensorNetworkModule(message_size, self.neuron_size)
        self.fc = nn.Linear(self.neuron_size + metric_size, 1)

    def forward(self, b_x, b_adj, a_x, a_adj, metrics):
        # change the design here. add adjacency matrix to graph convolution class so not pass it every time.
        b_node_embeddings = self.gnn14(self.gnn13(self.gnn12(self.gnn11(b_x, b_adj), b_adj), b_adj), b_adj)
        b_embedding = self.attention(b_node_embeddings[:-1, :]).flatten()
        a_node_embeddings = self.gnn24(self.gnn23(self.gnn22(self.gnn21(a_x, a_adj), a_adj), a_adj), a_adj)
        a_embedding = self.attention(a_node_embeddings[:-1, :]).flatten()
        # agg = torch.hstack([b_embedding, a_embedding])   # maybe a distance measure later
        agg = self.tensor_net(b_embedding, a_embedding).flatten()
        # features = torch.hstack([agg, metrics])
        features = agg
        output = self.fc(features)
        return output, agg

