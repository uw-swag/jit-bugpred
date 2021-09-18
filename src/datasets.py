import json
import os
import pandas as pd
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
    def __init__(self, data_dict, commit_lists, special_token=True, transform=None):
        self.transform = transform
        self.special_token = special_token
        self.data_dict = data_dict
        self.commit_lists = commit_lists
        with open(data_path + self.data_dict['labels']) as file:
            self.labels = json.load(file)
        self.ast_dict = None
        self.c_list = None
        self.file_index = 0
        self.mode = 'train'
        self.vectorizer_model = None
        self.learn_vectorizer()

    def learn_vectorizer(self):
        files = list(self.data_dict['train']) + list(self.data_dict['val'])
        corpus = []
        for fname in files:
            with open(data_path + fname) as fp:
                subtrees = json.load(fp)
            for commit, files in subtrees.items():
                for f in files:
                    for node_feat in f[1][0]:
                        if len(node_feat) > 1:  # None
                            corpus.append(node_feat)
                        else:
                            if not self.special_token:
                                corpus.append(node_feat[0])
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
                            if not self.special_token:
                                corpus.append(node_feat[0])
                            else:
                                feature = node_feat[0]
                                if ':' in node_feat[0]:
                                    feat_type = node_feat[0].split(':')[0]
                                    feature = feat_type + ' ' + '<' + feat_type[
                                                                      :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
                                corpus.append(feature)

        vectorizer = CountVectorizer(lowercase=False, max_features=400000,
                                     preprocessor=lambda x: x, binary=True)
        self.vectorizer_model = vectorizer.fit(corpus)

    def set_mode(self, mode):
        self.mode = mode
        self.c_list = pd.read_csv(data_path + self.commit_lists[self.mode])['commit_id'].tolist()
        self.file_index = 0
        with open(data_path + self.data_dict[self.mode][self.file_index], 'r') as fp:
            self.ast_dict = json.load(fp)

    def switch_datafile(self):
        self.file_index += 1
        self.file_index %= len(self.data_dict[self.mode])
        with open(data_path + self.data_dict[self.mode][self.file_index], 'r') as fp:
            self.ast_dict = json.load(fp)

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
        adj = sp.vstack([adj, np.ones((1, adj.shape[1]), dtype=np.float32)])
        adj = sp.hstack([adj, np.zeros((adj.shape[0], 1), dtype=np.float32)])
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def get_embedding(self, file_node_tokens, colors):
        for i, node_feat in enumerate(file_node_tokens):
            file_node_tokens[i] = node_feat.strip()
            if node_feat == 'N o n e':
                file_node_tokens[i] = 'None'
                colors.insert(i, 'blue')
                assert colors[i] == 'blue'
            if self.special_token:
                if ':' in node_feat:
                    feat_type = node_feat.split(':')[0]
                    file_node_tokens[i] = feat_type + ' ' + '<' + feat_type[
                                                                  :3].upper() + '>'  # e.g. number: 14 -> number <NUM>
        # fix the data later to remove the code above.
        features = self.vectorizer_model.transform(file_node_tokens).astype(np.float32)
        # add color feature at the end of features
        color_feat = [1 if c == 'red' else 0 for c in colors]
        features = sp.hstack([features, np.array(color_feat, dtype=np.float32).reshape(-1, 1)])
        # add supernode
        features = sp.hstack([features, np.zeros((features.shape[0], 1), dtype=np.float32)])
        supernode_feat = np.zeros((1, features.shape[1]), dtype=np.float32)
        supernode_feat[-1, -1] = 1
        features = sp.vstack([features, supernode_feat])
        features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        return features

    def __len__(self):
        return len(self.c_list)

    def __getitem__(self, item):
        c = self.c_list[item]
        while True:
            try:
                commit = self.ast_dict[c]
                break
            except:
                self.switch_datafile()
        label = self.labels[c]
        b_node_tokens, b_edges, b_colors = [], [[], []], []
        a_node_tokens, a_edges, a_colors = [], [[], []], []
        b_nodes_so_far, a_nodes_so_far = 0, 0
        for file in commit:
            b_node_tokens += [' '.join(node) for node in file[1][0]]
            b_colors += [c for c in file[1][2]]
            b_edges = [
                b_edges[0] + [s + b_nodes_so_far for s in file[1][1][0]],  # source nodes
                b_edges[1] + [d + b_nodes_so_far for d in file[1][1][1]]  # destination nodes
            ]
            a_node_tokens += [' '.join(node) for node in file[2][0]]
            a_colors += [c for c in file[2][2]]
            a_edges = [
                a_edges[0] + [s + a_nodes_so_far for s in file[2][1][0]],  # source nodes
                a_edges[1] + [d + a_nodes_so_far for d in file[2][1][1]]  # destination nodes
            ]

            b_n_nodes = len(file[1][0])
            a_n_nodes = len(file[2][0])
            b_nodes_so_far += b_n_nodes
            a_nodes_so_far += a_n_nodes

        before_embeddings = self.get_embedding(b_node_tokens, b_colors)
        before_adj = self.get_adjacency_matrix(b_nodes_so_far, b_edges[0], b_edges[1])
        after_embeddings = self.get_embedding(a_node_tokens, a_colors)
        after_adj = self.get_adjacency_matrix(a_nodes_so_far, a_edges[0], a_edges[1])
        training_data = [before_embeddings, before_adj, after_embeddings, after_adj, label]

        if not b_nodes_so_far and not a_nodes_so_far:
            print('commit {} has no file tensors.'.format(c))

        return training_data


if __name__ == "__main__":
    # ast_dataset = ASTDataset(data_path + '/asts_300_synerr.json')
    # print(ast_dataset[0])
    # train_loader = DataLoader(ast_dataset, batch_size=1, shuffle=False)
    # train_iter = iter(train_loader)
    # data = train_iter.next()
    print()
