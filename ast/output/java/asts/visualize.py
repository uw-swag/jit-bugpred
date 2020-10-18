from graphviz import Source

path = 'ast_0.dot'
s = Source.from_file(path)
s.view()


# from graphviz import Source
# temp = """
# digraph ___ast_code_model_py {
# 0 {1 2 3};
# 1 {4 5};
# 4 {6 7};
# 6 {};
# 7 {};
# 5 {};
# 2 {8 9 10 11};
# 8 {};
# 9 {};
# 10 {};
# 11 {12 13 14 15};
# 12 {};
# 13 {};
# 14 {16 17 18 19 20};
# 16 {};
# 17 {};
# 18 {21 22 23};
# 21 {};
# 22 {24 25 26};
# 24 {};
# 25 {};
# 26 {};
# 23 {};
# 19 {};
# 20 {27 28 29 30};
# 27 {};
# 28 {};
# 29 {31 32};
# 31 {33 34 35};
# 33 {36 37};
# 36 {};
# 37 {38 39};
# 38 {};
# 39 {};
# 34 {};
# 35 {};
# 32 {};
# 30 {};
# 15 {};
# 3 {};
# }
# """
# s = Source(temp, filename="test.gv", format="png")
# s.view()

import ast
from collections import defaultdict


class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.source = source  # lines of the source code

    def __str__(self):
        return str(self.graph)

    def _getid(self, node):
        try:
            lineno = node.lineno - 1
            return "%s: %s" % (type(node), self.source[lineno].strip())

        except AttributeError:
            return type(node)

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)

            elif isinstance(value, ast.AST):
                node_source = self._getid(node)
                value_source = self._getid(value)
                self.graph[node_source].append(value_source)
                # self.graph[type(node)].append(type(value))
                self.visit(value)


source = '''
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, train_inputs):
        self.train_inputs = train_inputs

        self.model = LogisticRegression(random_state=0)

'''

g = AstGraphGenerator(source)
tree = ast.parse(source)
g.generic_visit(tree)
print(ast.dump(tree))

# body = [
#     ImportFrom(module='sklearn.linear_model', names=[alias(name='LogisticRegression', asname=None)], level=0),
#     ClassDef(name='LogisticRegressionModel', bases=[], keywords=[],
#              body=[
#                  FunctionDef(name='__init__', args=arguments(
#                      args=[arg(arg='self', annotation=None), arg(arg='train_inputs', annotation=None)], vararg=None,
#                      kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
#                              body=[Assign(targets=[
#                                  Attribute(value=Name(id='self', ctx=Load()), attr='train_inputs', ctx=Store())],
#                                           value=Name(id='train_inputs', ctx=Load())),
#                                    Assign(targets=[
#                                        Attribute(value=Name(id='self', ctx=Load()), attr='model', ctx=Store())],
#                                           value=Call(func=Name(id='LogisticRegression', ctx=Load()), args=[],
#                                                      keywords=[keyword(arg='random_state', value=Num(n=0))]))],
#                              decorator_list=[], returns=None)],
#              decorator_list=[])]
