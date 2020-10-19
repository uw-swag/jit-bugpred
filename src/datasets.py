import json
import time

import pandas as pd
import numpy as np
import os
import requests
import subprocess

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class GitMiner:
    def __init__(self):
        self.base_url = 'https://api.github.com'
        with open(os.path.join(BASE_PATH, 'conf/auth.conf'), 'r') as file:
            lines = file.readlines()
        username = lines[0].split('\n')[0]
        password = lines[1].split('\n')[0]
        self.session = requests.Session()
        self.session.auth = (username, password)

    def get_before_after_content(self, commit_id):
        commit = self.search_commit(commit_id)
        headers = {'content-type': 'application/json',
                   'accept': 'application/vnd.github.v3+json'}
        response = self.session.get(commit.get('url'), headers=headers)
        commit = json.loads(response.text)
        changed_files = commit.get('files')
        filenames = [file.get('filename') for file in changed_files]
        before_contents = []
        after_contents = []
        headers['accept'] = 'application/vnd.github.v3.raw'
        for file in changed_files:
            response = self.session.get(file.get('contents_url'), headers=headers)
            after_contents.append(response.text)
            response = self.session.get(file.get('contents_url') + '&ref=' + commit.get('parents')[0].get('sha'),
                                        headers=headers)
            before_contents.append(response.text)

        return list(zip(filenames, before_contents, after_contents))

    def search_commit(self, commit_id):
        headers = {'content-type': 'application/json',
                   'accept': 'application/vnd.github.cloak-preview'}
        response = self.session.get(self.base_url + '/search/commits?q=' + commit_id, headers=headers)
        time.sleep(1)
        response_dict = json.loads(response.text)
        return response_dict.get('items')[0]

    def get_content(self, commit):
        repo_full_name = commit.get('repository').get('full_name')
        headers = {'content-type': 'application/json',
                   'accept': 'application/vnd.github.v3.raw'}
        response = self.session.get(self.base_url + '/repos/' + repo_full_name + '/contents/', headers=headers)

    # def find_commits(self):
    #     headers = {'content-type': 'application/json',
    #                'accept': 'application/vnd.github.cloak-preview'}
    #     keys = []
    #     ids = []
    #     messages = []
    #     for issue in self.issue_keys:
    #         print(issue)
    #         with open(join(path, 'auth.conf'), 'r') as file:
    #             lines = file.readlines()
    #         username = lines[0].split('\n')[0]
    #         password = lines[1].split('\n')[0]
    #         session = requests.Session()
    #         session.auth = (username, password)
    #         response = session.get(self.base_url + issue, headers=headers)
    #         time.sleep(1)
    #         response_dict = json.loads(response.text)
    #         commits = response_dict.get('items')
    #         for commit in commits:
    #             if not self.remove_modify_commit(commit):
    #                 keys.append(issue)
    #                 ids.append(commit.get('sha'))
    #                 messages.append(commit.get('commit').get('message'))
    #                 with open(join(data_dir, 'issue_commit.txt'), 'a') as file:
    #                     file.write(str(keys[-1]) + ', ' +
    #                                str(ids[-1]) + ', ' +
    #                                str(messages[-1]) + '\n')
    #                 print('\tfor', issue, ' commit', ids[-1], ' added.')
    #     commit_df = pd.DataFrame({'Issue key': keys,
    #                               'SHA': ids,
    #                               'Message': messages})
    #     df = commit_df.join(self.df.set_index('Issue key'), on='Issue key')
    #     return df
    #
    # @staticmethod
    # def remove_modify_commit(commit):
    #     headers = {'content-type': 'application/json',
    #                'accept': 'application/vnd.github.v3.patch'}
    #     url = commit.get('url')
    #     with open(join(path, 'auth.conf'), 'r') as file:
    #         lines = file.readlines()
    #     username = lines[0].split('\n')[0]
    #     password = lines[1].split('\n')[0]
    #     session = requests.Session()
    #     session.auth = (username, password)
    #     response = session.get(url, headers=headers)
    #     time.sleep(1)
    #     if response.text == '':
    #         return True
    #     text = response.text.split('\n')
    #     p = re.compile(', [0-9]+ deletion(s{0,1})\(-\)')
    #     for line in text:
    #         if bool(p.search(line)):
    #             return True
    #     return False


class ASTExtractor:
    pass


def get_tse():
    df = pd.read_csv(data_path + '/rawdata.csv')
    feature_columns = df.columns.tolist()[6:34] + df.columns.tolist()[35:]
    X = df[feature_columns].to_numpy()
    X = X.astype(float)
    y = df['buggy'].tolist()
    y = [int(label) for label in y]
    nans = list(set(np.argwhere(np.isnan(X))[:, 0]))
    X = np.delete(X, nans, axis=0)
    y = np.delete(y, nans, axis=0)

    return X[:11000], y[:11000], X[11000:], y[11000:]


def get_source_codes(s_index, e_index):
    df = pd.read_csv(data_path + '/rawdata.csv')
    commit_ids = df['commit_id'].tolist()[s_index:e_index]
    miner = GitMiner()
    contents = dict()
    for c in commit_ids:
        contents[c] = miner.get_before_after_content(c)
        print('commit', c, 'source codes fetched!')

    with open(data_path + '/source_codes_' + str(e_index) + '.json', 'w') as fp:
        json.dump(contents, fp)


def get_asts(filename):
    empty_template = '{"message":"Not Found",' \
                     '"documentation_url":"https://docs.github.com/rest/reference/repos#get-repository-content"}'
    supported_files = ['py', 'java', 'c', 'cpp']
    with open(os.path.join(data_path, filename), 'r') as fp:
        commit_codes = json.load(fp)
    for commit, codes in commit_codes.items():
        for c in codes:
            fname = c[0].split('/')[-1]
            if c[1] == empty_template or not fname.split('.')[-1] in supported_files:
                continue
            if not os.path.exists(os.path.join(BASE_PATH, 'ast', 'code', commit)):
                os.makedirs(os.path.join(BASE_PATH, 'ast', 'code', commit))
            with open(os.path.join(BASE_PATH, 'ast', 'code', commit, c[0].split('/')[-1]), 'w') as fp:
                fp.write(c[1])

    command = './cli.sh parse --lang py,java,c,cpp --project ' + \
              os.path.join(BASE_PATH, 'ast', 'code') + \
              ' --output ' + os.path.join(BASE_PATH, 'ast', 'output') + \
              ' --storage dot'
    print(command)
    subprocess.run(command, cwd=os.path.join(BASE_PATH, 'astminer-master-dev'), shell=True)


if __name__ == "__main__":
    # get_source_codes(0, 300)
    get_asts('source_codes_300.json')
