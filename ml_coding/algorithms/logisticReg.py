import numpy as np

"""
This is pretty similar to linearReg
"""

class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_iteration=5000):
        self.learning_rate_ = learning_rate
        self.max_iteration_ = max_iteration

    def fit(self, X, y):
        """
        sigmoid(X*W) = y, where W is the parameter to be fitted
        :param X: 2D array
        :param y:
        :return:
        """
        n_row, n_feature = X.shape
        self.W =  np.ones(shape=(n_feature + 1, 1)) # add constant -- interception
        X = self.pad_ones(X)
        for _ in range(self.max_iteration_):
            self.update_weights(X, y)

    def pad_ones(self, X):
        ones = np.ones(X.shape[0]).reshape(-1,1)
        X = np.concatenate([X, ones], axis=1)
        return X

    def predict_prob(self, X):
        X = self.pad_ones(X)
        return self._predict(X)


    def predict(self, X):
        r = self.predict_prob(X)
        r = r.flatten()
        mask_0 = r < 0.5
        r[mask_0] = 0
        r[~mask_0] = 1
        r = r.astype(int)
        return r

    def _predict(self, X, delta=None):
        """

        :param X: The shape 1 of X equal self.W.shape[1]
        :param delta: Used to change self.W
        :return:
        """
        if delta is None:
            linear = np.matmul(X, self.W)
        else:
            linear = np.matmul(X, self.W + delta)
        return self.sigmoid(linear)

    def get_cost(self, X:np.array, y:np.array, delta):
        """
        :param X: 2D
        :param y: 1D
        :param delta: 1D of length X.shape[1]
        :return:
        """
        if X.shape[1] < len(self.W):
            X = self.pad_ones(X)
        preds = self._predict(X, delta=delta)
        preds = preds.flatten()
        y = y.flatten()
        # print(preds)
        zero_cost = - (1-y) * np.log2(1 + 1e-9 - preds)
        one_cost = - y * np.log2(1e-9 + preds)
        return np.sum(zero_cost) + np.sum(one_cost)


    def get_gradient(self, X, y):
        gradient = np.zeros(len(self.W))
        for loc in range(len(self.W)):
            gradient[loc] = self.get_gradient_loc(X, y, loc)
        return gradient.reshape(-1, 1)

    def get_gradient_loc(self, X, y, loc):
        """
        [COST(W+small;X) - COST(W-small;X)] / (2 * small)
        :param X:
        :param y:
        :param loc:
        :return:
        """
        small = 1e-5
        delta_pos = np.zeros(self.W.shape)
        delta_neg = np.zeros(self.W.shape)
        delta_pos[loc, 0] = small
        delta_neg[loc, 0] = -small
        delta_cost = self.get_cost(X,y,delta_pos) - self.get_cost(X,y,delta_neg)
        return delta_cost / (2 * small)




    def sigmoid(self, x):
        return np.exp(-x) / (1 + np.exp(-x))

    def update_weights(self, X, y):
        gradient = self.get_gradient(X, y)
        self.W = self.W - self.learning_rate_ * gradient