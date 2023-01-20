import numpy as np
from ml_coding.algorithms.naivebayes import NaiveBayes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_naive_bayes():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    model = NaiveBayes()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    print(preds)
    assert  score == 1.0


# if __name__ == "__main__":
#     test_naive_bayes()