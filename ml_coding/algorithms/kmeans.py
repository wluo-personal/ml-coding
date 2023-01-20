import numpy as np

class KMeans:

    def __init__(self, K, max_iteration, stop=1.0):
        self.K_ = K
        self.max_iteration_ = max_iteration
        self.stop_ = stop
        self.SMALL_ = 1e-5

    def assign_new_centroid(self, X, assignment):
        """
        calculate the new centroid given current assignment
        :param X: original data to be fitted
        :param assignment: 1D array of length X.shape[0], where the value
            is any in (0, K), which indicate the which centroid the X[i] belongs
            to
        :return:
            new centroid
        """
        centroids = []
        for value in range(self.K_):
            mask = assignment == value
            centroids.append(X[mask].mean(axis=0))
        return np.array(centroids)


    def calculate_error(self, X, centroids):
        """

        :param X:
        :param centroids:
        :return: matrix with same shape <X.shape[0], K>
        """
        errors = []
        for each in centroids:
            error = np.sum((X - each) ** 2, axis=1)
            errors.append(error)
        errors = np.array(errors).T
        return errors


    def fit(self, X):
        previous_error = np.inf
        # 1. init
        centroids = new_centroid = self.init_centroids(X)
        self.ERROR_VALUE_ = []

        # 2.
        for _ in range(self.max_iteration_):
            errors = self.calculate_error(X, new_centroid)
            error_value = errors.min(axis=1).sum()
            if previous_error - error_value <= self.SMALL_ :
                break
            else:
                previous_error = error_value
                self.ERROR_VALUE_.append(error_value)
                centroids = new_centroid
                # assign new center based on error
                cluster_assignment = errors.argmin(axis=1)
                # assign new centroid
                new_centroid = self.assign_new_centroid(X, assignment=cluster_assignment)
        self.centroids = centroids

    def init_centroids(self, X):
        centroids = []
        for idx in range(X.shape[1]):
            min_, max_ = X[:, idx].min(), X[:, idx].max()
            centroids.append(
                np.random.uniform(low=min_, high=max_, size=self.K_)
            )
        centroids = np.array(centroids)
        return centroids.T

    def transform(self, X):
        """
        :param X: 2D array <# of records, features>
        :return: 1D array
        """
        errors = self.calculate_error(X, self.centroids)
        return np.argmin(errors, axis=1)
