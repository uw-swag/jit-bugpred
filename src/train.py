import os

from datasets import get_tse
from models import LogisticRegressionModel

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trained_models')
    train_inputs, train_labels, test_inputs, test_labels = get_tse()
    lrm = LogisticRegressionModel(train_inputs, train_labels, test_inputs, test_labels, save_dir, '/logreg.model')
    lrm.train()
    print('model trained!')
