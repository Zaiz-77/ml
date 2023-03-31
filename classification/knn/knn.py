import numpy as np


def create_dataset():
    train = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    labels = ['Love', 'Love', 'Love', 'Fight', 'Fight', 'Fight']
    return train, labels


def classify(inx, dataset, labels, k):
    n, most = dataset.shape[0], 0
    dis = (((np.tile(inx, (n, 1)) - dataset) ** 2).sum(axis=1)) ** 0.5
    idx = dis.argsort()
    cnt = {}
    for i in range(k):
        label = labels[idx[i]]
        cnt[label] = cnt.get(label, 0) + 1
        most = max(most, cnt[label])
    for x in cnt:
        if cnt[x] == most:
            return x

# def classify(inx, dataset, labels, k):
#     date_size = dataset.shape[0]
#     diff_mat = np.tile(inx, (date_size, 1)) - dataset
#     sq_diff_mat = diff_mat ** 2
#     sq_distances = sq_diff_mat.sum(axis=1)
#     distances = sq_distances ** 0.5
#     sorted_distances = distances.argsort()
#     class_count = {}
#     for i in range(k):
#         label = labels[sorted_distances[i]]
#         class_count[label] = class_count.get(label, 0) + 1
#     res = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
#     return res[0][0]


def main():
    train, labels = create_dataset()
    print(classify([80, 5], train, labels, 3))


if __name__ == '__main__':
    main()
