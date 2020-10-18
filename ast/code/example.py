from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, train_inputs):
        self.train_inputs = train_inputs
        self.model = LogisticRegression(random_state=0)

