import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter  #for vote in knn
#style.use('fivethirtyeight')
import pandas as pd
import random

# dataset = {'k': [[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}

# new_features = [5,7]

# for i in dataset:
#    for ii in dataset[i]
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1])
# plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('actually it\'s totally ok!')
    dist = []
    var2 = []
    var4 = []

    for group in data:
        # print(group)
        for features in data[group]:
            # euclid_dist = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            euclid_dist = np.linalg.norm(np.array(features)-np.array(predict))
            dist.append([euclid_dist, group])

            if group == 2:
                var2.append(np.array(features))

            if group == 4:
                var4.append(np.array(features))


    # print('var2 shape in ',np.shape(var2))
    # print('var4 shape in ',np.shape(var4))

            # print(euclid_dist)
            # print(len(euclid_dist))
            # print(dist)
    votes = [i[1] for i in sorted(dist)[:k]]
    # print(votes)
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


# result = k_nearest_neighbors(dataset,new_features, 5)

# print(result)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
col_names = ['id', 'clump', 'uni_cell_size', 'uni_cell_shape', 'marg_adh', 'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'normal_nuclei','mitoses','class']
df = pd.read_csv(url, header=None, names=col_names)
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# print(df.head())
full_data = df.astype(float).values.tolist()
# print(full_data[:5])
random.shuffle(full_data)  # shuffle data
#
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1]) # class is the last field

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, 7)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy = ', correct/total)