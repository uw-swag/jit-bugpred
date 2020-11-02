import ast
import json
import os

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


if __name__ == "__main__":
    ast_dict = get_asts('source_codes_1000.json')
    with open(data_path + '/asts_1000.json', 'w') as fp:
        json.dump(ast_dict, fp)
