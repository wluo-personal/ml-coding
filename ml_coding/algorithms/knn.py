import numpy as np

class KNN:
    def __init__(self, K=10):
        self.K_ = K

    def calculate_distance(self, X):
        """

        :param X: 1D array <features>
        :return: 1D array, length is training data length
        """
        diff = (self.X_ - X)
        distance = np.sum(diff ** 2, axis=1) ** 0.5
        return distance


    def fit(self, X, y):
        """

        :param X: 2D array <records, features>
        :param y: 1D array <category>
        :return:
        """
        self.X_ = X
        self.y_ = y

    def _get_most_frequenct(self, array):
        counts = {}
        answer = (0, None)
        for each in array:
            counts[each] = counts.get(each, 0) + 1
            if counts[each] > answer[0]:
                answer = (counts[each], each)
        return counts, answer[-1]

    def predict(self, X):
        """
        :param X: 1D array <# of features>
        :return:
        """
        distance = self.calculate_distance(X)
        args = np.argsort(distance)[:self.K_]
        return self._get_most_frequenct(self.y_[args])

