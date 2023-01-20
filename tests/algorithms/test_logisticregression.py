import numpy as np
from ml_coding.algorithms.logisticReg import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_decision_tree():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    mask = y < 2
    X = X[mask]
    y = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    print(preds)
    assert  score == 1.0

# if __name__ == "__main__":
#     test_decision_tree()