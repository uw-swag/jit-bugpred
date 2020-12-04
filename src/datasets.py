import ast
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

from ast_visitor import ASTVisitor

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 768


def get_asts(filenames, start):
    empty_template = '{"message":"Not Found",' \
                     '"documentation_url":"https://docs.github.com/rest/reference/repos#get-repository-content"}'
    supported_files = ['py']
    ast_dict = dict()
    n_commits = 0

    for filename in filenames:
        with open(os.path.join(data_path, filename), 'r') as fp:
            commit_codes = json.load(fp)

        for commit, files in commit_codes.items():
            for f in files:  # f is a tuple (name, before content, after content)
                fname = f[0].split('/')[-1]  # path/to/«file.py»
                ftype = fname.split('.')[-1]
                # exclude newly added files and unsupported ones
                if f[1] == empty_template or ftype not in supported_files:
                    continue

                if commit not in ast_dict:
                    ast_dict[commit] = [(f[0],)]
                else:
                    ast_dict[commit].append((f[0],))

                before_ast = 'SYNTAX ERROR'
                try:
                    b_visitor = ASTVisitor()
                    b_tree = ast.parse(f[1])
                    b_visitor.visit(b_tree)
                    before_ast = b_visitor.get_ast()
                except:
                    print(commit, f[0], 'before')

                ast_dict[commit][-1] += (before_ast,)

                after_ast = 'SYNTAX ERROR'
                try:
                    a_visitor = ASTVisitor()
                    a_tree = ast.parse(f[2])
                    a_visitor.visit(a_tree)
                    after_ast = a_visitor.get_ast()
                except:
                    print(commit, f[0], 'after')

                ast_dict[commit][-1] += (after_ast,)

            n_commits += 1

            if n_commits == 500:
                print(len(ast_dict))
                start += n_commits
                with open(data_path + '/asts_' + str(start) + '_synerr.json', 'w') as fp:
                    json.dump(ast_dict, fp)
                print('/asts_' + str(start) + '_synerr.json saved.')
                ast_dict = dict()
                n_commits = 0

    print(len(ast_dict))
    start += n_commits
    with open(data_path + '/asts_' + str(start) + '_synerr.json', 'w') as fp:
        json.dump(ast_dict, fp)
    print('/asts_' + str(start) + '_synerr.json saved.')


class ASTDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.transform = transform
        df = pd.read_csv(data_path + '/rawdata.csv')
        df = df[['commit_id', 'buggy']]
        df.set_index('commit_id', inplace=True)
        self.label_dict = df.to_dict()['buggy']
        with open(filename, 'r') as fp:
            ast_dict = json.load(fp)
        self.ast_dict = list(ast_dict.items())
        print('let\'s do it locally')
        self.tokenizer = AutoTokenizer.from_pretrained(data_path + '/codebert', local_files_only=True)
        self.codebert = AutoModel.from_pretrained(data_path + '/codebert', output_hidden_states=True, local_files_only=True)
#        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
#        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base", output_hidden_states=True)

        self.codebert.to(device)

    @staticmethod
    def get_adjacency_matrix(n_nodes, src, dst):
        # note that adjacency matrices in this problem are very sparse
        a = torch.zeros(n_nodes, n_nodes)
        for i in range(len(src)):
            a[src[i], dst[i]] = 1
        return a

    def get_embedding(self, file_node_tokens):
        # more alternatives at https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        # maybe not efficient
        # will initial_representation participate in BP?
        self.codebert.eval()
        with torch.no_grad():
            initial_representations = torch.zeros(len(file_node_tokens), HIDDEN_SIZE)
            for i, node_token in enumerate(file_node_tokens):
                hidden_states = self.codebert(self.tokenizer
                                              .encode(node_token, return_tensors='pt', max_length=512)
                                              .to(device))[2]
                try:
                    initial_representations[i, :] = torch.mean(hidden_states[-1], dim=1).squeeze()
                except RuntimeError as e:
                    print(e)

        return initial_representations

    def __len__(self):
        return len(self.ast_dict)

    def __getitem__(self, item):
        commit = self.ast_dict[item]
        label = 1 if self.label_dict[commit[0]] else 0
        training_data = []
        if len(commit[1]) > 5:
            return None
        for file in commit[1]:
            # file is a list of 3 elements: name, before, and after. before and after are lists of two things
            # node tokens and node edges. node tokens is a list of lists of node tokens (node -> type + token)
            if isinstance(file[1], str):    # for SYNTAX ERROR cases
                continue

            try:
                b_n_nodes = max(max(file[1][1][0]), max(file[1][1][1])) + 1
                a_n_nodes = max(max(file[2][1][0]), max(file[2][1][1])) + 1
            except ValueError:
                print(file[0], 'skipped -> 0 nodes!')
                continue

            if b_n_nodes > 5000 or a_n_nodes > 5000:
                continue

            before_tokens = self.get_embedding([' '.join(node) for node in file[1][0]])
            after_tokens = self.get_embedding([' '.join(node) for node in file[2][0]])
            before_adj = self.get_adjacency_matrix(b_n_nodes, file[1][1][0], file[1][1][1])
            after_adj = self.get_adjacency_matrix(a_n_nodes, file[2][1][0], file[2][1][1])
            training_data.append([before_tokens, before_adj, after_tokens, after_adj, label])

        if self.transform is not None:
            training_data = self.transform(training_data)

        return training_data


if __name__ == "__main__":
    get_asts(['source_codes_200.json'], 0)
    print('Python 3 ASTs saved.')
    # with open(data_path + '/asts_300_synerr.json', 'r') as fp:
    #     ast_dict = json.load(fp)

    # ast_dataset = ASTDataset(data_path + '/asts_300_synerr.json')
    # print(ast_dataset[0])
    # train_loader = DataLoader(ast_dataset, batch_size=1, shuffle=False)
    # train_iter = iter(train_loader)
    # data = train_iter.next()
    # print()
