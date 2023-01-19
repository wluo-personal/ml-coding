import numpy as np

"""
step 1:
    initialize W -> Y = X * W. Note padding X with ones to last dim so that the 
        last dim in W will be the interception
step 2:
    Get Gradient:
    Pay attention to calculate derivative with respect to W.
    Option 1: Calculate by math
    Option 2: (COST(W+h;X) - COST(W-h;X)) / (2h)
        Here the 
step 3:
    Gradient Decent:
    W = W - learning_rate * gradient


"""

class LinearReg:
    """

    reference
    https://www.askpython.com/python/examples/linear-regression-from-scratch
    """
    def __init__(self, learning_rate, iterations):
        self.learning_rate_ = learning_rate
        self.iterations_ = iterations

    def fit(self, X, y):
        self.shape_ = X.shape
        # add one-dim for constant
        self.W =  np.ones(self.shape_[1] + 1).reshape(-1,1)
        self.COSTS_ = []
        for _ in range(self.iterations_):
            self.update_weight(X, y)
            self.COSTS_.append(self.get_cost(X, y))

    def pad_ones(self, X):
        return np.concatenate(
            [X, np.ones(X.shape[0]).reshape(-1,1)], axis=-1)

    def predict(self, X):
        X = self.pad_ones(X)
        return self._predict(X)

    def _predict(self, X):
        return np.matmul(X, self.W)


    def get_cost(self, X, y):
        """
        l2 cost
        """
        pred = self.predict(X)
        return np.matmul((y-pred).T, (y-pred)) / (2 * self.shape_[0])

    def get_gradient(self, X, y):
        """
        take deravative with respect to W.
        Cost =  1/2N * [y - h(x;w)]^2
        Simplified: Cost = 1/2N * (y - wx)^2
            d(Cost) / d(w) = 1/N * (x^2*w - y*x)
                            = 1/N * x (xw - y)
                            = 1/N * x (h - y)
                                where h is the prediction, y is label
        """
        X = self.pad_ones(X)
        return 1 / self.shape_[0] * np.matmul(X.T, self._predict(X) - y)

    def update_weight(self, X, y):
        gradient = self.get_gradient(X, y)
        self.W = self.W - self.learning_rate_ * gradient


class LinearRegV2:
    """
    This version is for automatic gradient calculation
    reference
    https://www.askpython.com/python/examples/linear-regression-from-scratch
    """
    def __init__(self, learning_rate, iterations):
        self.learning_rate_ = learning_rate
        self.iterations_ = iterations
        self.SMALL = 1

    def fit(self, X, y):
        self.shape_ = X.shape
        # add one-dim for constant
        self.W =  np.ones(self.shape_[1] + 1).reshape(-1,1)
        self.COSTS_ = []
        for _ in range(self.iterations_):
            self.update_weight(X, y)
            self.COSTS_.append(self.get_cost(X, y))

    def pad_ones(self, X):
        return np.concatenate(
            [X, np.ones(X.shape[0]).reshape(-1,1)], axis=-1)

    def predict(self, X, delta=None):
        X = self.pad_ones(X)
        return self._predict(X, delta)

    def _predict(self, X, delta=None):
        if delta is None:
            return np.matmul(X, self.W)
        else:
            return np.matmul(X, self.W+delta)


    def get_cost(self, X, y, delta=None):
        """
        l2 cost
        """
        pred = self.predict(X, delta)
        return np.matmul((y-pred).T, (y-pred)) / (2 * self.shape_[0])

    def get_gradient(self, X, y):
        """
        take deravative with respect to W.
        Cost =  1/2N * [y - h(x;w)]^2
        Simplified: Cost = 1/2N * (y - wx)^2
            d(Cost) / d(w) = 1/N * (x^2*w - y*x)
                            = 1/N * x (xw - y)
                            = 1/N * x (h - y)
                                where h is the prediction, y is label
        """
        deravative = np.zeros(self.W.shape)
        for loc in range(0, deravative.shape[0]):
            deravative[loc] = self.get_loc_gradient(X, y, loc)
        return deravative

    def get_loc_gradient(self, X, y, loc):
        delta_pos = np.zeros(self.W.shape)
        delta_neg = np.zeros(self.W.shape)
        delta_pos[loc,0] = self.SMALL
        delta_neg[loc,0] = -self.SMALL
        return (
                self.get_cost(X,y,delta_pos) - self.get_cost(X,y,delta_neg)
        ) / (2 * self.SMALL)


    def update_weight(self, X, y):
        gradient = self.get_gradient(X, y)
        self.W = self.W - self.learning_rate_ * gradient
