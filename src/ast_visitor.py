import ast
from _ast import AST


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


if __name__ == "__main__":

    source = '''
from sklearn.linear_model import LogisticRegression
    
    
class LogisticRegressionModel(LogisticRegression):
    def __init__(self, train_inputs):
        self.train_inputs = train_inputs
    
        self.model = LogisticRegression(random_state=0)
    '''

    tree = ast.parse(source)
    c = ASTVisitor()
    c.visit(tree)
    p = c.get_ast()
    print()
