import numpy as np
from ml_coding.algorithms.kmeans import KMeans

def gen_data():
    center = [np.array((2, 0, 0, 0)), np.array((0, 2, 0, 0)),
              np.array((0, 0, 2, 0))]
    data = []
    for each_center in center:
        data.append(each_center + np.random.normal(0, 0.7, size=(100, 4)))
    data = np.concatenate(data, axis=0)
    return center, data
    # data shape is (300, 4)

def test_kmeans():
    km = KMeans(K=3, max_iteration=5000)
    center, data = gen_data()
    km.fit(data)
    print(f"ori center {center} \n fitted center: {km.centroids}")

    for each_ori_center in center:
        exists = False
        for each_fit_center in km.centroids:
            diff = each_ori_center - each_fit_center
            if np.abs(np.max(diff)) < 0.2:
                exists = True
                break
        if not exists:
            raise ValueError(f"The center {each_ori_center} is not close "
                             f"to any fitted center.")

if __name__ == "__main__":
    test_kmeans()

