import json
import time
import pandas as pd
import os
import requests
from pydriller import RepositoryMining, ModificationType

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_PATH, 'data')
MAX_N_CHANGED_FILES = 3


class GitMiner:
    def __init__(self):
        self.base_url = 'https://api.github.com'
        # self.max_n_files = max_n_changed_files
        with open(os.path.join(BASE_PATH, 'conf', 'auth.conf'), 'r') as file:
            lines = file.readlines()
        self.token = lines[2].split('\n')[0]
        self.session = requests.Session()

    def get_before_after_content(self, commit_id):
        try:
            commit = self.search_commit(commit_id)
        except IndexError:
            return None
        headers = {'Authorization': 'token ' + self.token,
                   'content-type': 'application/json',
                   'accept': 'application/vnd.github.v3+json'}
        response = self.session.get(commit.get('url'), headers=headers)
        commit = json.loads(response.text)
        changed_files = commit.get('files')
        # if len(changed_files) > self.max_n_files:
        #     return None
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
        headers = {'Authorization': 'token ' + self.token,
                   'content-type': 'application/json',
                   'accept': 'application/vnd.github.cloak-preview'}
        query = self.base_url + '/search/commits?q=' + commit_id + '+org:openstack'
        response = self.session.get(query, headers=headers)
        time.sleep(2)
        response_dict = json.loads(response.text)
        return response_dict.get('items')[0]


def get_source_codes():
    df = pd.read_csv(data_path + '/alltrain.csv')
    collected = pd.concat([pd.read_csv(data_path + '/ballowfiletrain.csv'),
                           pd.read_csv(data_path + '/ballowfileval.csv'),
                           pd.read_csv(data_path + '/ballowfiletest.csv')])
    df = df[~df['commit_id'].isin(collected['commit_id'])]
    # df_dict = df[['commit_id', 'buggy']].sample(frac=1).set_index('commit_id').to_dict()['buggy']
    # df = df[df['buggy']][['commit_id', 'buggy']].sample(frac=1)     # SELECT commit_id, buggy WHERE buggy='TRUE';

    miner = GitMiner()

    contents = dict()
    # buggy_cntr = {'True': 0, 'False': 0}
    # ratio = 0.25
    for i, row in df.iterrows():
        cmtid = row['commit_id']
        # if not buggy and buggy_cntr['True'] < buggy_cntr['False'] * ratio:
        # continue
        content = miner.get_before_after_content(cmtid)
        if content is None:
            print('\t\tcommit', cmtid, 'skipped!')
            continue
        contents[cmtid] = content
        # buggy_cntr[str(buggy)] += 1
        print('commit', cmtid, 'source codes fetched!')

    with open(data_path + '/source_codes_' + 'alltrain' + '.json', 'w') as fp:
        json.dump(contents, fp)
    print('\nfinished.')
    # print('buggy counter:', buggy_cntr)


def get_project_name():
    df = pd.read_csv(data_path + '/rawdata.csv', dtype={'revd': str, 'buggy': str, 'fix': str})
    miner = GitMiner()

    projects = []
    for i, row in df.iterrows():
        cmtid = row['commit_id']
        proj = None
        while not proj:
            try:
                response = miner.search_commit(cmtid)
                proj = response['repository']['full_name']
            except IndexError:
                print('\t\t*****', cmtid)
                proj = 'unkowncommit'
            except TypeError:
                time.sleep(30)

        if i % 50 == 49:
            print('a batch of 50 commits finished.')
        projects.append(proj)
        assert i == len(projects) - 1

    df = df.assign(project=pd.Series(projects).values)
    df.to_csv('newrawdata.csv', index=False)
    print('\nfinished.')


def update_n_files():
    df = pd.read_csv(data_path + '/newrawdata.csv', dtype={'revd': str, 'buggy': str, 'fix': str})
    projects = ['https://github.com/' + p + '.git' for p in df['project'].unique() if p != 'unkowncommit']
    repo_mining = RepositoryMining(projects, only_commits=df['commit_id'].tolist())
    nfs = dict()
    for i, c in enumerate(repo_mining.traverse_commits()):
        nf = 0
        for m in c.modifications:
            if m.filename.endswith('.py') and m.change_type is ModificationType.MODIFY:
                nf += 1
        nfs[c.hash] = nf
        if i % 50 == 49:
            print('a batch of 50 commits finished.')

    nf_list = []
    for i, row in df.iterrows():
        if row['project'] == 'unkowncommit':
            nf = row['nf']
        else:
            nf = nfs[row['commit_id']]
        nf_list.append(nf)

    df['nf'] = nf_list
    df.to_csv(data_path + '/newrawdata.csv', index=False)
    print(nf_list[:30])
    print('\nfinished.')


if __name__ == "__main__":
    # get_source_codes()
    get_project_name()
