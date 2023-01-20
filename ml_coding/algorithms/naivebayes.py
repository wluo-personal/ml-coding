import numpy as np
from collections import Counter, defaultdict

"""
Assumption: each feature is i.i.d, if it is numeric, it is subject to Gaussian 
    distribution

P(class | X) -- posterior 
= P (X | class) * P (class) / P(X)

1. P(X | class) is the likelihood
    = P(feature1=v1, feature2=v2, feature3=v3 ... | class)
    = P(feature1=v1 | class) * P(feature2=v2 | class) ... because of i.i.d

2. P(class) -- prior

3. P(X) -- marginal probability
    = P(feature1=v1) * P(feature2=v2)...

"""

class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        # 1. check categorical and Numerical features
        self._set_feature_type(X)

        # 2. fit prior
        self._fit_prior(y)

        # 3. fit marginal probability
        self._fit_marginal_probability(X)

        # 4. fit likelihood
        self._fit_likelihood(X, y)

    def predict(self, X):
        """

        :param X: 2D array <records, features>
        :return:
        """
        result = []
        for each in X:
            result.append(self._predict(each))
        return np.array(result)

    def _predict(self, x):
        """

        :param x: 1D array - features
        :return:
        """

        marginal = self._get_marginal_probability(x)
        max_posterior = 0.0
        max_class = None
        for class_name in self.prior:
            prior = self._get_prior(class_name)
            likelihood = self._get_likelihood(x, class_name)
            posterior = prior * likelihood / marginal
            if posterior > max_posterior:
                max_posterior = posterior
                max_class = class_name
        return max_class



    def _fit_marginal_probability(self, X):
        """
        This will fit marignal probability. This method will create a dictionary
        where key is feature_id, value is a dictionary or tuple.

        (1)if the feature type is category, the value will be a dictionary where
        key is the posible values, value is the probability
        (2)if the feature type is numeric, the value will be a tuple of two
        (mean, std)
        :return:
        """
        self.marginal = {}
        for idx in range(X.shape[1]):
            if self.feature_is_numeric[idx]:
                avg = X[:,idx].mean()
                std = X[:,idx].std()
                self.marginal[idx] = (avg, std)
            else:
                dict_ = {}
                for value in X[:,idx].unique():
                    dict_[value] = np.sum(X[:,idx] == value) / X.shape[0]
                self.marginal[idx] = dict_

    def _fit_likelihood(self, X, y):
        """
        This is the most complicated fit. Pay attention to this method.
        :param X: 2D array <records, features>
        :param y: 1D array, length of records -- label
        :return:
        It will create a dictionary to store the result
        self.likelihood:  dictionary, key - label posible values, value -
        dictionary.
            For the value dictionary, the key is feature index, value can be
            tuple (if the feature is numeric) or dictionary if the value is
            categorical <posible values: probability>
        """
        self.likelihood = {}
        for class_value in np.unique(y):
            mask_class = y == class_value
            X_class = X[mask_class]
            class_dict = {}
            for idx in range(X.shape[1]):
                if self.feature_is_numeric[idx]:
                    avg = X_class[:, idx].mean()
                    std = X_class[:, idx].std()
                    class_dict[idx] = (avg, std)
                else:
                    feature_dict = {}
                    for feature_value in X_class[:,idx].unique():
                        feature_dict[feature_value] = np.sum(
                            X_class[:,idx] == feature_value
                        ) / X_class.shape[0]
                    class_dict[idx] = feature_dict
            self.likelihood[class_value] = class_dict

    def _fit_prior(self, y):
        """
        fit prior. This will create a dictionary self.prior to store the result
        in self.prior:
            key - posible values
            value - probability
        :param y:
        :return:
        """
        counter = Counter(y)
        self.prior = {} # key<posible values>: value<probability>
        for value, freq in counter.items():
            self.prior[value] = freq / len(y)

    def _gaussian_pdf(self, x, avg, std):
        exponential = np.exp(-0.5 * ((x-avg)/std)**2)
        prod = 1 / (((2*np.pi) ** 0.5) * std)
        return prod * exponential

    def _get_prior(self, class_name):
        return self.prior[class_name]

    def _get_likelihood(self, x, class_name):
        prob = 1.0
        class_dict = self.likelihood[class_name]
        for idx in range(len(x)):
            feature_value = x[idx]
            if self.feature_is_numeric[idx]:
                avg, std = class_dict[idx]
                prob *= self._gaussian_pdf(feature_value, avg, std)
            else:
                prob *= class_dict[idx][feature_value]
        return prob

    def _get_marginal_probability(self, x):
        """

        :param x: 1D array where the length = # of features
        :return:
        """
        prob = 1.0
        for idx in range(len(x)):
            feature_value = x[idx]
            if self.feature_is_numeric[idx]:
                avg, std = self.marginal[idx]
                prob *= self._gaussian_pdf(feature_value, avg, std)
            else:
                prob *= self.marginal[idx][feature_value]
        return prob



    def _set_feature_type(self, X):
        """

        :param X: 2D array. <records, features>
        :return:
        """
        self.feature_is_numeric = {}
        for idx in range(X.shape[1]):
            if X[:, 1].dtype == np.dtype("float"):
                self.feature_is_numeric[idx] = True
            else:
                self.feature_is_numeric[idx] = False

