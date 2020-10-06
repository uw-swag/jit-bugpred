import pandas as pd
import numpy as np
import os

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


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


if __name__ == "__main__":
    print()

