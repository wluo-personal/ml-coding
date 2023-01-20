import numpy as np
from collections import Counter
"""
For this implementation, there are several crucial parts
1. Recursive -- To build subtrees
2. This decision tree only support classification
3. How to decide split? 
    Support Gini gain and information gain
4. Keep in mind:
    leaf node is used to give a prediction
    node is used to make a decision
"""
class Node:
    def __init__(
            self,
            feature=None,
            threshold=None,
            left=None,
            right=None,
            value=None):
        """

        :param feature: Int, the index of the feature
        :param threshold: number
        :param left: left tree. If X[feature] <= threshold
        :param right: right tree. If X[feature] > threshold
        :param value: None - for node, some value for leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class Gain:
    def __init__(self, method="gini"):
        assert method in ("gini", "entropy", "information gain")
        if method == "gini":
            self._calc_purity= self._get_gini
        else:
            self._calc_purity = self._get_entropy

    def _get_entropy(self, Y):
        N = len(Y)
        counts = np.bincount(np.array(Y).astype(int))
        entropy = 0
        # print(counts)
        for c in counts:
            p = c / N
            # 1e-9 is used to avoid log2(0)
            entropy -= p * np.log2(p + 1e-9)
        return entropy

    def _get_gini(self, Y):
        N = len(Y)
        counts = np.bincount(np.array(Y).astype(int))
        gini = 1
        for c in counts:
            p = c / N
            gini = gini - p**2
        return gini

    def get_gain(self, Y_parent, Y_left, Y_right):
        """ This is the information gain"""
        purity_parent = self._calc_purity(Y_parent)
        purity_left = self._calc_purity(Y_left)
        purity_right = self._calc_purity(Y_right)
        gain = purity_parent - \
               len(Y_left)/len(Y_parent) * purity_left - \
               len(Y_right)/len(Y_parent) * purity_right
        return gain







class DecisionTree:
    def __init__(
            self,
            max_depth=5,
            min_sample_split=2,
            split_method="entropy"):
        self.max_depth_ = max_depth
        self.min_sample_split_ = min_sample_split
        self.gain_method = Gain(split_method)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        gain = 0
        data = {}
        for feature_id in range(n_features):
            for thred in np.unique(X[:,feature_id]):
                mask_left = X[:,feature_id] <= thred
                y_left, y_right = y[mask_left], y[~mask_left]
                split_gain = self.gain_method.get_gain(y,y_left,y_right)
                if split_gain > gain:
                    gain = split_gain
                    data = {
                        "gain": gain,
                        "X_left": X[mask_left],
                        "y_left": y_left,
                        "X_right": X[~mask_left],
                        "y_right": y_right,
                        "feature": feature_id,
                        "threshold": thred
                    }
        return data

    def _build_tree(self, X, y, depth=0):
        n_row, n_col = X.shape
        if n_row >= self.min_sample_split_ and depth <= self.max_depth_:
            # tree can be grown
            split_data = self._best_split(X, y)
            if len(split_data) > 0:
                root = Node(
                    feature=split_data["feature"],
                    threshold=split_data["threshold"],
                    left=self._build_tree(
                        X=split_data["X_left"],
                        y=split_data["y_left"],
                        depth=depth+1),
                    right=self._build_tree(
                        X=split_data["X_right"],
                        y=split_data["y_right"],
                        depth=depth+1),
                )
                return root
        ### leaf node
        node = Node(value=Counter(y).most_common(1)[0][0])
        return node


    def fit(self, X, y):
        """

        :param X: 2D array
        :param y: 1D array
        :return:
        """
        self.root = self._build_tree(X,y)

    def predict(self, X):
        """
        return the category it belongs to
        :param X: 2D array
        :return: np.array
        """
        result = []
        for x in X:
            result.append(self._predict(x, self.root))
        return np.array(result)

    def _predict(self, x, tree: Node):
        if tree.value is not None:
            return tree.value
        else:
            if x[tree.feature] <= tree.threshold:
                return self._predict(x, tree.left)
            else:
                return self._predict(x, tree.right)