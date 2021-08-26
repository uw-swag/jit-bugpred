import json
import pandas as pd
import numpy as np

train = pd.read_csv('data/unbalance_train.csv')['commit_id']
valid = pd.read_csv('data/unbalance_valid.csv')['commit_id']

files = ['subtrees_apachejava_color_1.json', 'subtrees_apachejava_color_2.json',
         'subtrees_apachejava_color_3.json', 'subtrees_apachejava_color_4.json',
         'subtrees_cleanapachejava_color_1.json', 'subtrees_cleanapachejava_color_2.json',
         'subtrees_cleanapachejava_color_3.json', 'subtrees_cleanapachejava_color_4.json',
         'subtrees_cleanapachejava_color_5.json', 'subtrees_cleanapachejava_color_6.json',
         'subtrees_cleanapachejava_color_7.json']

valid_ast = dict()
test_ast = dict()

for f in files:
    with open('data/' + f) as fp:
        asts = json.load(fp)
#     for id in test:
#         if id in asts:
#             test_ast[id] = asts[id]
#     print('test finished.')
    for id in valid:
        if id in asts:
            valid_ast[id] = asts[id]
#     print('valid finished.')
    print('file finished.')

# print(len(test_ast))
# with open('data/balance_test.json', 'w') as fp:
#     json.dump(test_ast, fp)

print(len(valid_ast))
with open('data/unbalance_valid.json', 'w') as fp:
    json.dump(valid_ast, fp)

print('test and valid finished.')

size = 15000
for i in range((len(train) // size) + 1):
    train_ast = dict()
    for f in files:
        with open('data/' + f) as fp:
            asts = json.load(fp)
        for id in train[i*size:(i+1)*size]:
            if id in asts:
                train_ast[id] = asts[id]
        print('switching file ...')
    print('ast size: {}'.format(len(train_ast)))
    with open('data/unbalance_train_{}.json'.format(i+1), 'w') as fp:
        json.dump(train_ast, fp)
    print('written on file, next bucket ...')


# keys = pd.read_csv('data/keys_apachejava_ast.csv')
# print(len(keys))
# keys = keys.drop_duplicates()
# print(len(keys))
# keys = keys['commit_id']
#
# files = ['subtrees_apachejava_color_1.json', 'subtrees_apachejava_color_2.json',
#          'subtrees_apachejavanew_color_1.json', 'subtrees_apachejavanew_color_2.json']
#
# size = 10000
# for i in range((len(keys) // size) + 1):
#     newast = dict()
#     for f in files:
#         with open('data/' + f) as fp:
#             asts = json.load(fp)
#         for id in keys[i*size:(i+1)*size]:
#             if id in asts:
#                 newast[id] = asts[id]
#         print('switching file ...')
#     print('ast size: {}'.format(len(newast)))
#     with open('data/subtrees_apachejava_new_{}.json'.format(i+1), 'w') as fp:
#         json.dump(newast, fp)
#     print('written on file, next bucket ...')
