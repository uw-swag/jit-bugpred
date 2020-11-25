from _ast import AST
import ast
import json
import os

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class ASTVisitor:
    def __init__(self):
        self.nodes = list()  # list of nodes
        self.ast_features = dict()  # maps a node id to list of its features
        self.ast_graph = dict()  # maps a node id to list of node ids to which it has arrows

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
            if (i + 1) in self.ast_features:
                features[i] += self.ast_features[(i + 1)]
        # edges = [[source nodes], [destination nodes]]
        edges = [[], []]
        for k, v in self.ast_graph.items():
            for node in v:
                edges[0].append(k - 1)
                edges[1].append(node - 1)

        return features, edges


def get_asts(ast_filename, content_filename):

    with open(os.path.join(data_path, ast_filename), 'r') as fp:
        ast_dict = json.load(fp)

    with open(os.path.join(data_path, content_filename), 'r') as fp:
        commit_dict = json.load(fp)

    to_delete = []
    for commit, files in ast_dict.iteritems():
        for i in xrange(len(files)):
            index = -1
            if files[i][1] == 'SYNTAX ERROR':
                index = next((j for j, v in enumerate(commit_dict[commit]) if v[0] == files[i][0]), None)
                code = commit_dict[commit][index][1]
                b_visitor = ASTVisitor()
                try:
                    b_tree = ast.parse(code)
                    b_visitor.visit(b_tree)
                    before_ast = b_visitor.get_ast()
                    files[i] = (files[i][0], before_ast, files[i][2])
                except Exception as e:
                    try:
                        b_tree = ast.parse(code.split('\n', 2)[2:][0])
                        b_visitor.visit(b_tree)
                        before_ast = b_visitor.get_ast()
                        files[i] = (files[i][0], before_ast, files[i][2])
                    except Exception as e:
                        print commit, files[i][0], 'before'
                        print e
                        to_delete.append(commit)
                        break

            if files[i][2] == 'SYNTAX ERROR':
                if index == -1:
                    index = next((j for j, v in enumerate(commit_dict[commit]) if v[0] == files[i][0]), None)
                code = commit_dict[commit][index][2]
                a_visitor = ASTVisitor()
                try:
                    a_tree = ast.parse(code)
                    a_visitor.visit(a_tree)
                    after_ast = a_visitor.get_ast()
                    files[i] = (files[i][0], files[i][1], after_ast)
                except Exception as e:
                    try:
                        a_tree = ast.parse(code.split('\n', 2)[2:][0])
                        a_visitor.visit(a_tree)
                        after_ast = a_visitor.get_ast()
                        files[i] = (files[i][0], files[i][1], after_ast)
                    except Exception as e:
                        print commit, files[i][0], 'after'
                        print e
                        to_delete.append(commit)
                        break

    for c in to_delete:
        ast_dict.pop(c)

    return ast_dict


if __name__ == "__main__":
#     source = '''
# def ali():
#     print "Hello world"
#     '''
#
#     tree = ast.parse(source)
#     c = ASTVisitor()
#     c.visit(tree)
#     p = c.get_ast()
#     print "\n"
    ast_dict = get_asts('asts_600_synerr.json', 'source_codes_600.json')
    print 'asts fetched!'
    # thanks to
    # https://stackoverflow.com/questions/25203209/how-to-fix-json-dumps-error-utf8-codec-cant-decode-byte-0xe0-in-position-2
    with open(data_path + '/asts_600.json', 'w') as fp:
        fp.write(json.dumps(ast_dict, encoding='latin1'))
