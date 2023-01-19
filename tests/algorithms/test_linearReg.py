import numpy as np
from ml_coding.algorithms.linearReg import LinearRegV2

def gen_data_set():
    func = lambda x: 5.0 * x[0] - 3 * x[1] + 2 * x[2] - 7 * x[3] + 3.2
    X = np.random.randint(low=-100, high=100, size=(1000, 4)).astype(float)
    W = np.array([5.0, -3, 2, -7]).reshape(-1, 1)
    y = np.matmul(X, W) + 3.2
    return X, y

def test_linearReg():
    X,y = gen_data_set()
    lin = LinearRegV2(learning_rate=0.0001, iterations=5000)
    lin.fit(X,y)
    W = lin.W
    assert abs(W[0, 0] - 5.0) < 0.02
    assert abs(W[1, 0] + 3.0) < 0.02
    assert abs(W[2, 0] - 2.0) < 0.02
    assert abs(W[3, 0] + 7.0) < 0.02

# if __name__ == "__main__":
#     test_linearReg()
