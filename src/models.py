from sklearn.linear_model import LogisticRegression
import pickle


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


import torch
import torch.nn as nn


class Propagator(nn.Module):
    """
    Gated Propagator for GGNN
    Using LSTM gating mechanism
    """

    def __init__(self, state_dim, n_node):
        super(Propagator, self).__init__()

        self.n_node = n_node

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node:]
        A_out = A[:, :, self.n_node:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, state_dim, annotation_dim, n_node, n_steps):
        super(GGNN, self).__init__()

        assert state_dim >= annotation_dim, 'state_dim must be no less than annotation_dim'

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_node = n_node
        self.n_steps = n_steps

        # incoming and outgoing edge embedding
        self.in_edge = nn.Linear(self.state_dim, self.state_dim)
        self.out_edge = nn.Linear(self.state_dim, self.state_dim)

        # Propagation Model
        self.propagator = Propagator(self.state_dim, self.n_node)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()      # make sure weights are set after this and we don't need .apply()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_state = self.in_edge(prop_state)
            out_state = self.out_edge(prop_state)
            in_state = torch.transpose(in_state, 0, 1).contiguous()
            in_state = in_state.view(-1, self.n_node, self.state_dim)
            out_state = torch.transpose(out_state, 0, 1).contiguous()
            out_state = out_state.view(-1, self.n_node, self.state_dim)

            prop_state = self.propagator(in_state, out_state, prop_state, A)

        # I guess here I should use stack because we expect third dim here but I removed stack from in_states
        # and now our tensors have two dims.
        join_state = torch.stack((prop_state, annotation))
        # join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        output = output.sum(2)

        return output
