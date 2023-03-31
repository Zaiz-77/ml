from math import log


def create_dataset():
    train = [
        [1, 1, 'Yes'],
        [1, 1, 'yes'],
        [1, 0, 'No'],
        [0, 1, 'No'],
        [0, 1, 'No']
    ]
    labels = ["No surfacing", "Flippers"]
    return train, labels


def calc_shannon(dataset):
    n = len(dataset)
    cnt = {}
    for v in dataset:
        label = v[-1]
        cnt[label] = cnt.get(label, 0) + 1
    shannon = 0.0
    for label in cnt:
        prob = float(cnt[label]) / n
        shannon -= prob * log(prob, 2)
    return shannon


def split_dataset(dataset, axis, value):
    sub_dataset = []
    for record in dataset:
        if record[axis] == value:
            prev = record[:axis]
            prev.extend(record[axis + 1:])
            sub_dataset.append(prev)
    return sub_dataset


def get_best_feat(dataset):
    n = len(dataset[0]) - 1
    base = calc_shannon(dataset)
    mx_gain, idx = 0.0, -1
    for i in range(n):
        feat_set = set([v[i] for v in dataset])
        nxt_shannon = 0.0
        for value in feat_set:
            subset = split_dataset(dataset, i, value)
            prob = len(subset) / len(dataset)
            nxt_shannon += prob * calc_shannon(subset)
        dealt = base - nxt_shannon
        if dealt > mx_gain:
            mx_gain = dealt
            idx = i
    return idx


def majority_count(feat):
    counter = {}
    for f in feat:
        counter[f] = counter.get(f, 0) + 1
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return sorted_counter[0][0]


def create_tree(dataset, labels):
    feats = [d[-1] for d in dataset]
    if feats.count(feats[0]) == len(feats):
        return feats[0]
    if len(dataset[0]) == 1:
        return majority_count(feats)
    feat = get_best_feat(dataset)
    label = labels[feat]
    res = {label: {}}
    del(labels[feat])
    subset = set([d[feat] for d in dataset])
    for v in subset:
        sub_labels = labels[:]
        res[label][v] = create_tree(split_dataset(dataset, feat, v), sub_labels)
    return res


def classify(tree, labels, vec):
    res = ""
    root = list(tree.keys())[0]
    fi, idx = tree[root], labels.index(root)
    for decision in fi.keys():
        if decision == vec[idx]:
            if type(fi[decision]).__name__ == 'dict':
                res = classify(fi[decision], labels, vec)
            else:
                res = fi[decision]
    return res


def main():
    dataset, labels = create_dataset()
    tree = create_tree(dataset, labels.copy())
    print(classify(tree, labels, [1, 0]))


if __name__ == '__main__':
    main()
