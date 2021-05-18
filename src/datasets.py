import json
import os

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 768


class ASTDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.data_dict = data_dict
        with open(data_path + self.data_dict['labels']) as file:
            self.labels = json.load(file)
        self.ast_dict = None
        self.vectorizer_model = None
        self.learn_vectorizer()

    def learn_vectorizer(self):
        with open(data_path + self.data_dict['train']) as file:
            tr_subtrees = json.load(file)
        with open(data_path + self.data_dict['val']) as file:
            va_subtrees = json.load(file)
        subtrees = {**tr_subtrees, **va_subtrees}
        assert len(subtrees) == len(tr_subtrees) + len(va_subtrees)

        corpus = []
        for commit, files in subtrees.items():
            for f in files:
                for node_feat in f[1][0]:
                    if len(node_feat) > 1:  # None
                        corpus.append(node_feat)
                    else:
                        feature = node_feat[0]
                        if ':' in node_feat[0]:
                            feat_type = node_feat[0].split(':')[0]
                            feature = feat_type + ' ' + '<' + feat_type[
                                                              :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
                        corpus.append(feature)
                for node_feat in f[2][0]:
                    if len(node_feat) > 1:  # None
                        corpus.append(node_feat)
                    else:
                        feature = node_feat[0]
                        if ':' in node_feat[0]:
                            feat_type = node_feat[0].split(':')[0]
                            feature = feat_type + ' ' + '<' + feat_type[
                                                              :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
                        corpus.append(feature)

        vectorizer = CountVectorizer(binary=True)
        self.vectorizer_model = vectorizer.fit(corpus)

    def set_mode(self, mode):
        with open(data_path + self.data_dict[mode], 'r') as fp:
            ast_dict = json.load(fp)
        self.ast_dict = list(ast_dict.items())

    @staticmethod
    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    @staticmethod
    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_adjacency_matrix(self, n_nodes, src, dst):
        edges = np.array([src, dst]).T
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(n_nodes, n_nodes),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # add supernode
        adj = sp.vstack([adj, sp.coo_matrix(np.ones((1, adj.shape[1]), dtype=np.float32))])
        adj = sp.hstack([adj, sp.coo_matrix(np.zeros((adj.shape[0], 1), dtype=np.float32))])
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def get_embedding(self, file_node_tokens):
        if file_node_tokens[0] == 'N o n e':
            file_node_tokens[0] = 'None'
        for i, node_feat in enumerate(file_node_tokens):
            if ':' in node_feat:
                feat_type = node_feat.split(':')[0]
                file_node_tokens[i] = feat_type + ' ' + '<' + feat_type[:3].upper() + '>'  # e.g. number: 14 -> number <NUM>
        # fix the data later to remove the code above.
        features = self.vectorizer_model.transform(file_node_tokens).astype(np.float32)
        # add supernode
        features = sp.hstack([features, sp.csr_matrix(np.zeros((features.shape[0], 1), dtype=np.float32))])
        supernode_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
        supernode_feat[-1, -1] = 1
        features = sp.vstack([features, sp.csr_matrix(supernode_feat)])
        features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        return features

    def __len__(self):
        return len(self.ast_dict)

    def __getitem__(self, item):
        commit = self.ast_dict[item]
        label = self.labels[commit[0]]
        training_data = []
        for file in commit[1]:
            b_n_nodes = len(file[1][0])
            a_n_nodes = len(file[2][0])

            before_tokens = self.get_embedding([' '.join(node) for node in file[1][0]])
            after_tokens = self.get_embedding([' '.join(node) for node in file[2][0]])
            before_adj = self.get_adjacency_matrix(b_n_nodes, file[1][1][0], file[1][1][1])
            after_adj = self.get_adjacency_matrix(a_n_nodes, file[2][1][0], file[2][1][1])
            training_data.append([before_tokens, before_adj, after_tokens, after_adj, label])

        if not len(training_data):
            print('commit {} has no file tensors.'.format(commit[0]))
        return training_data


if __name__ == "__main__":
    ast_dataset = ASTDataset(data_path + '/asts_300_synerr.json')
    print(ast_dataset[0])
    # train_loader = DataLoader(ast_dataset, batch_size=1, shuffle=False)
    # train_iter = iter(train_loader)
    # data = train_iter.next()
    print()
