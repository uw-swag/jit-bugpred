import json
import time
import pandas as pd
import os
import requests

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')


class GitMiner:
    def __init__(self):
        self.base_url = 'https://api.github.com'
        with open(os.path.join(BASE_PATH, 'conf', 'auth.conf'), 'r') as file:
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


if __name__ == "__main__":
    get_source_codes(600, 1000)
