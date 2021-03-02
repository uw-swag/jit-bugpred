import json
import os
import numpy as np
import pandas as pd

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')

with open(os.path.join(data_path, 'subtrees_0.25_3.json'), 'r') as fp:
    subtrees = json.load(fp)

df = pd.read_csv(data_path + '/rawdata.csv')
df = df[['commit_id', 'buggy']]
df.set_index('commit_id', inplace=True)
label_dict = df.to_dict()['buggy']
subtree_list = list(subtrees.items())
n_buggy = len([cid for cid, _ in subtree_list if label_dict[cid]])

np.random.shuffle(subtree_list)

with open(os.path.join(data_path, 'subtrees_0.25_val.json'), 'w') as file:
    json.dump(dict([(k, v) for k, v in subtree_list[:400]]), file)

with open(os.path.join(data_path, 'subtrees_0.25_test.json'), 'w') as file:
    json.dump(dict([(k, v) for k, v in subtree_list[400:800]]), file)

with open(os.path.join(data_path, 'subtrees_0.25_train.json'), 'w') as file:
    json.dump(dict([(k, v) for k, v in subtree_list[800:]]), file)

