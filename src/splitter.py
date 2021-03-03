import json
import os
import numpy as np
import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

with open(os.path.join(data_path, 'subtrees_0.25_3.json'), 'r') as fp:
    subtrees = json.load(fp)

# zero_nodes = dict()
# all_f_counts = 0
# for commit, files in subtrees.items():
#     f_count = 0
#     for f in files:
#         if len(f[1][0]) == 0 or len(f[2][0]) == 0:
#             f_count += 1
#     if f_count:
#         zero_nodes[commit] = f_count
#         all_f_counts += f_count
#
# subtrees_cp = dict()
# for commit, files in subtrees.items():
#     new_files = []
#     for f in files:
#         if len(f[1][0]) == 0 and len(f[2][0]) == 0:
#             continue
#         elif len(f[1][0]) == 0:
#             f[1][0].append('None')
#         elif len(f[2][0]) == 0:
#             f[2][0].append('None')
#         new_files.append(f)
#
#     if len(new_files):
#         subtrees_cp[commit] = new_files

df = pd.read_csv(data_path + '/rawdata.csv')
df = df[['commit_id', 'buggy']]
df.set_index('commit_id', inplace=True)
label_dict = df.to_dict()['buggy']
subtree_list = list(subtrees_cp.items())
n_buggy = len([cid for cid, _ in subtree_list if label_dict[cid]])

np.random.shuffle(subtree_list)

with open(os.path.join(data_path, 'subtrees_0.25_val.json'), 'w') as file:
    json.dump(dict([(k, v) for k, v in subtree_list[:350]]), file)

with open(os.path.join(data_path, 'subtrees_0.25_test.json'), 'w') as file:
    json.dump(dict([(k, v) for k, v in subtree_list[350:700]]), file)

with open(os.path.join(data_path, 'subtrees_0.25_train.json'), 'w') as file:
    json.dump(dict([(k, v) for k, v in subtree_list[700:]]), file)

