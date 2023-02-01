import numpy as np
import random
import copy
import mld_metrics


def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def distances_one_all(sample_idx, elements_idxs, x):
    distances = []
    for elem_idx in elements_idxs:
        distances.append((elem_idx, calculate_distance(x[sample_idx, :], x[elem_idx, :])))

    return distances

def new_sample(sample, ref_neighbor, neighbors, x, y):
    synth_sample = np.zeros(x.shape[1])

    for feature_idx in range(len(synth_sample)):
        # missing to add when feature is not numeric. In that case it should
        # put the most frequent value in the neighbors
        diff = x[ref_neighbor, feature_idx] - x[sample, feature_idx]
        offset = diff * random.random() #* .5
        value = x[sample, feature_idx] + offset

        synth_sample[feature_idx] = value

    labels_counts = y[sample, :]
    labels_counts = np.add(labels_counts, np.sum(y[neighbors, :], axis=0)) # todo 这里？
    labels = labels_counts > (len(neighbors) + 1) / 2
    ori_label = y[sample]
    return synth_sample, labels, ori_label

def sortSamples(v):
    return v[1]

def MLSMOTE(x, y, k):
    mean_ir = mld_metrics.mean_ir(y)

    y_new = copy.deepcopy(y)
    x_new = copy.deepcopy(x)
    y_new_ori = copy.deepcopy(y)
    min_set = set()
    all_bag =  list(range(len(y)))
    for label in range(y.shape[1]): # 所有可能的 样本类别
        ir_label,_ = mld_metrics.ir_per_label(label, y) # 最多类别数量 / 每个类别数量
        if ir_label > mean_ir:
            label_samples = y[:, label] == 1
            minority_bag = np.where(label_samples)[0] # 挑选对应label的样本

            label_max = y[:, label] != 1

            majority_bag = np.where(label_max)[0] # 挑选对应label的样本

            for sample_idx in minority_bag:
                min_set.add(sample_idx)
                distances = distances_one_all(sample_idx, all_bag, x_new)
            
                distances.sort(key=sortSamples)

                # ignore the first one since it'll be sample
                neighbors = [v[0] for v in distances[1:k+1]]
                ref_neighbor = random.sample(neighbors, k=1)

                synth_sample, labels, ori_label = new_sample(sample_idx, ref_neighbor, neighbors, x_new, y_new)

                x_new = np.append(x_new, [synth_sample], axis=0)
                y_new = np.append(y_new, [labels.astype(int)], axis=0)
                y_new_ori = np.append(y_new_ori, [ori_label.astype(int)], axis=0)

    #print (len(min_set))
    #return x_new[x.shape[0]:], y_new[y.shape[0]:],  y_new_ori[y.shape[0]:], min_set
    return x_new, y_new,  y_new_ori[y.shape[0]:], min_set



'''
def MLSMOTE(x, y, k):
    mean_ir = mld_metrics.mean_ir(y)

    y_new = copy.deepcopy(y)
    x_new = copy.deepcopy(x)
    y_new_ori = copy.deepcopy(y)
    min_set = set()

    for label in range(y_new.shape[1]): # 所有可能的 样本类别
        ir_label = mld_metrics.ir_per_label(label, y_new) # 最多类别数量 / 每个类别数量
        if ir_label > mean_ir:
            label_samples = y_new[:, label] == 1
            minority_bag = np.where(label_samples)[0] # 挑选对应label的样本

            for sample_idx in minority_bag:
                min_set.add(sample_idx)
                distances = distances_one_all(sample_idx, minority_bag, x_new)
            
                distances.sort(key=sortSamples)

                # ignore the first one since it'll be sample
                neighbors = [v[0] for v in distances[1:k+1]]
                ref_neighbor = random.sample(neighbors, k=1)

                synth_sample, labels, ori_label = new_sample(sample_idx, ref_neighbor, neighbors, x_new, y_new)

                x_new = np.append(x_new, [synth_sample], axis=0)
                y_new = np.append(y_new, [labels.astype(int)], axis=0)
                y_new_ori = np.append(y_new_ori, [ori_label.astype(int)], axis=0)

    print (len(min_set))
    return x_new[x.shape[0]:], y_new[y.shape[0]:],  y_new_ori[y.shape[0]:], min_set

'''

from sklearn.metrics.pairwise import pairwise_distances

def FUZZY_FILTER(X, y, X_samp, y_smote, target):
    X_min = X[y == target]
    X_maj = X[y != target]
    new_synth = []
    d = len(X[0])
    pos_cache = pairwise_distances(X_min, X_maj, metric='l1')
    pos_cache = 1.0 - pos_cache
    pos_cache = pos_cache.clip(0, d)
    pos_cache = 1.0 - pos_cache

    result_synth = []
    result_maj = []
    iteration = 0


    gamma_S = 0.7
    gamma_M = 0.03

    pos_synth = pairwise_distances(X_min, X_samp, metric='l1')
    pos_synth = 1.0 - pos_synth
    pos_synth = pos_synth.clip(0, d)
    pos_synth = 1.0 - pos_synth

    min_pos = np.min(pos_synth, axis=0)
    to_add = np.where(min_pos < gamma_S)[0]
    result_synth.extend(X_samp[to_add])
    new_synth.extend(X_samp[to_add])

    # checking the minimum POS values of the majority samples
    min_pos = np.min(pos_cache, axis=0)
    to_remove = np.where(min_pos < gamma_M)[0]




def MLSMOTE_FUZZYFILTER(x, y, k):
    mean_ir = mld_metrics.mean_ir(y)

    y_new = copy.deepcopy(y)
    x_new = copy.deepcopy(x)

    for label in range(y_new.shape[1]):  # 所有可能的 样本类别
        ir_label = mld_metrics.ir_per_label(label, y_new)  # 最多类别数量 / 每个类别数量
        if ir_label > mean_ir:
            label_samples = y_new[:, label] == 1
            minority_bag = np.where(label_samples)[0]  # 挑选对应label的样本
            X_labelsmote = []
            Y_labelsmote = []
            for sample_idx in minority_bag:
                distances = distances_one_all(sample_idx, minority_bag, x_new)

                distances.sort(key=sortSamples)

                # ignore the first one since it'll be sample
                neighbors = [v[0] for v in distances[1:k + 1]]
                ref_neighbor = random.sample(neighbors, k=1)

                synth_sample, labels = new_sample(sample_idx, ref_neighbor, neighbors, x_new, y_new)

                y_new = np.append(y_new, [labels.astype(int)], axis=0)
                x_new = np.append(x_new, [synth_sample], axis=0)
                Y_labelsmote.append(labels.astype(int))
                X_labelsmote.append(synth_sample)
            z = 1
            FUZZY_FILTER(x, y , X_labelsmote, Y_labelsmote, label)



    return x_new[x.shape[0]:], y_new[y.shape[0]:]


# Example of usage
'''
x, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=8)

print('Positive samples per class:')
print(np.sum(y, axis=0))

# Send the labels and the percentage to delete
nx, ny, oy, _ = MLSMOTE(x, y, 5)

print('Synthetic samples generated (count): ')
print(ny.shape[0])

print('Positive samples generated per class: ')
print(np.sum(ny, axis=0))
'''
