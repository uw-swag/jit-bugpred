import ast
import json
import os
from _ast import AST

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class ASTVisitor:
    def __init__(self):
        self.nodes = list()         # list of nodes
        self.ast_features = dict()  # maps a node id to list of its features
        self.ast_graph = dict()     # maps a node id to list of node ids to which it has arrows

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        # set id of root
        if len(self.nodes) == 0:
            self.nodes.append(node)
        node_id = len(self.nodes)
        # iterate over node features
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    # if the feature is a node
                    if isinstance(item, AST):
                        self.nodes.append(item)
                        child_id = len(self.nodes)
                        try:
                            self.ast_graph[node_id].append(child_id)
                        except KeyError:
                            self.ast_graph[node_id] = [child_id]
                        self.visit(item)
                    # keep feature if it's not a node
                    elif isinstance(item, str) and item != '':
                        try:
                            self.ast_features[node_id].append(item)
                        except KeyError:
                            self.ast_features[node_id] = [item]
            elif isinstance(value, AST):
                self.nodes.append(value)
                child_id = len(self.nodes)
                try:
                    self.ast_graph[node_id].append(child_id)
                except KeyError:
                    self.ast_graph[node_id] = [child_id]
                self.visit(value)
            # keep feature if it's not a node
            elif isinstance(value, str) and value != '':
                try:
                    self.ast_features[node_id].append(value)
                except KeyError:
                    self.ast_features[node_id] = [value]

    def get_ast(self):
        # features = [[node_type, node_token (opt.)], ...]
        features = [[n.__class__.__name__] for n in self.nodes]
        for i in range(len(self.nodes)):
            if (i+1) in self.ast_features:
                features[i] += self.ast_features[(i+1)]
        # edges = [[source nodes], [destination nodes]]
        edges = [[], []]
        for k, v in self.ast_graph.items():
            for node in v:
                edges[0].append(k-1)
                edges[1].append(node-1)

        return features, edges


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

            # if n_commits == 500:
            #     print(len(ast_dict))
            #     start += n_commits
            #     with open(data_path + '/asts_' + str(start) + '_synerr.json', 'w') as fp:
            #         json.dump(ast_dict, fp)
            #     print('/asts_' + str(start) + '_synerr.json saved.')
            #     ast_dict = dict()
            #     n_commits = 0

    print(len(ast_dict))
    start += n_commits
    with open(data_path + '/asts_' + str(start) + '_synerr.json', 'w') as fp:
        json.dump(ast_dict, fp)
    print('/asts_' + str(start) + '_synerr.json saved.')


if __name__ == "__main__":

#     source = '''
# from sklearn.linear_model import LogisticRegression
#
#
# class LogisticRegressionModel(LogisticRegression):
#     def __init__(self, train_inputs):
#         self.train_inputs = train_inputs
#
#         self.model = LogisticRegression(random_state=0)
#     '''
#
#     tree = ast.parse(source)
#     c = ASTVisitor()
#     c.visit(tree)
#     p = c.get_ast()
#     print()

    get_asts(['source_codes_0.25.json'], 0)
    print('Python 3 ASTs saved.')
    # with open(data_path + '/asts_300_synerr.json', 'r') as fp:
    #     ast_dict = json.load(fp)
