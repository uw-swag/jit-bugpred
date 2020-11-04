import ast
import json
import os
import numpy as np
from torch.utils.data import DataLoader

from ast_visitor import ASTVisitor

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


def get_asts(filename):
    empty_template = '{"message":"Not Found",' \
                     '"documentation_url":"https://docs.github.com/rest/reference/repos#get-repository-content"}'
    supported_files = ['py']

    with open(os.path.join(data_path, filename), 'r') as fp:
        commit_codes = json.load(fp)

    ast_dict = dict()
    for commit, files in commit_codes.items():
        for f in files:                     # f is a tuple (name, before content, after content)
            fname = f[0].split('/')[-1]     # path/to/«file.py»
            ftype = fname.split('.')[-1]
            # exclude newly added files and unsupported ones
            if f[1] == empty_template or ftype not in supported_files:
                continue

            try:
                b_visitor = ASTVisitor()
                b_tree = ast.parse(f[1])
                b_visitor.visit(b_tree)
                before_ast = b_visitor.get_ast()
            except Exception as e:
                print(commit, 'before')
                print(e)
                print()
                continue

            try:
                a_visitor = ASTVisitor()
                a_tree = ast.parse(f[2])
                a_visitor.visit(a_tree)
                after_ast = a_visitor.get_ast()
            except Exception as e:
                print(commit, 'after')
                print(e)
                print()
                continue

            ast_dict[commit] = [before_ast, after_ast]

    return ast_dict


def load_graphs_from_file(file_name):
    data_list = []
    edge_list = []
    target_list = []
    with open(file_name, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                data_list.append([edge_list,target_list])
                edge_list = []
                target_list = []
            else:
                digits = []
                line_tokens = line.split(" ")
                if line_tokens[0] == "?":
                    for i in range(1, len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    target_list.append(digits)
                else:
                    for i in range(len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    edge_list.append(digits)
    return data_list


def find_max_edge_id(data_list):
    max_edge_id = 0
    for data in data_list:
        edges = data[0]
        for item in edges:
            if item[1] > max_edge_id:
                max_edge_id = item[1]
    return max_edge_id


def find_max_node_id(data_list):
    max_node_id = 0
    for data in data_list:
        edges = data[0]
        for item in edges:
            if item[0] > max_node_id:
                max_node_id = item[0]
            if item[2] > max_node_id:
                max_node_id = item[2]
    return max_node_id


def find_max_task_id(data_list):
    max_node_id = 0
    for data in data_list:
        targe = data[1]
        for item in targe:
            if item[0] > max_node_id:
                max_node_id = item[0]
    return max_node_id


def split_set(data_list):
    n_examples = len(data_list)
    idx = range(n_examples)
    train = idx[:50]
    val = idx[-50:]
    return np.array(data_list)[train], np.array(data_list)[val]


def data_convert(data_list, n_annotation_dim):
    n_nodes = find_max_node_id(data_list)
    n_tasks = find_max_task_id(data_list)
    task_data_list = []
    for i in range(n_tasks):
        task_data_list.append([])
    for item in data_list:
        edge_list = item[0]
        target_list = item[1]
        for target in target_list:
            task_type = target[0]
            task_output = target[-1]
            annotation = np.zeros([n_nodes, n_annotation_dim])
            annotation[target[1]-1][0] = 1
            task_data_list[task_type-1].append([edge_list, annotation, task_output])
    return task_data_list


def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] =  1
    return a


class bAbIDataset():
    """
    Load bAbI tasks for GGNN
    """
    def __init__(self, path, question_id, is_train):
        all_data = load_graphs_from_file(path)
        self.n_edge_types = find_max_edge_id(all_data)
        self.n_tasks = find_max_task_id(all_data)
        self.n_node = find_max_node_id(all_data)

        all_task_train_data, all_task_val_data = split_set(all_data)

        if is_train:
            all_task_train_data = data_convert(all_task_train_data, 1)
            self.data = all_task_train_data[question_id]
        else:
            all_task_val_data = data_convert(all_task_val_data, 1)
            self.data = all_task_val_data[question_id]

    def __getitem__(self, index):
        am = create_adjacency_matrix(self.data[index][0], self.n_node, self.n_edge_types)
        annotation = self.data[index][1]
        target = self.data[index][2] - 1
        return am, annotation, target

    def __len__(self):
        return len(self.data)


class bAbIDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(bAbIDataLoader, self).__init__(*args, **kwargs)


if __name__ == "__main__":
    ast_dict = get_asts('source_codes_1000.json')
    with open(data_path + '/asts_1000.json', 'w') as fp:
        json.dump(ast_dict, fp)
