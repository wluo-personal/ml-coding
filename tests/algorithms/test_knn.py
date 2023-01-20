import numpy as np
from ml_coding.algorithms.knn import KNN

def gen_data():
    center = [np.array((2, 0, 0, 0)), np.array((0, 2, 0, 0)),
              np.array((0, 0, 2, 0))]
    data = []
    Y = []
    for idx, each_center in enumerate(center):
        data.append(each_center + np.random.normal(0, 0.7, size=(100, 4)))
        Y.append(np.ones(100) * idx)
    data = np.concatenate(data, axis=0)
    Y = np.concatenate(Y)
    return data, Y

def test_kmeans():
    knn = KNN(K=10)
    data, Y = gen_data()
    knn.fit(data, Y)

    test_data_1 = np.array([2.5, 0.3, 0.2, -0.3])
    test_data_2 = np.array([2.2, 0.2, 2.5, -0.3])
    test_data_3 = np.array([1.2, 0.2, 2.5, -0.3])

    res1 = knn.predict(test_data_1)
    res2 = knn.predict(test_data_2)
    res3 = knn.predict(test_data_3)
    assert res1[-1] == 0
    assert res3[-1] == 2
    print(f"The result of res2: {res2}")


# if __name__ == "__main__":
#     test_kmeans()

