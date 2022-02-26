import time
from collections import defaultdict
import numpy as np
import random
import queue
import scipy.sparse as sp
import copy
import networkx as nx

def save_candidates(filename, candidates):
    with open(filename, 'w') as T:
        for key in candidates.keys():
            T.write(str(key) + '\t')
            for i in candidates[key]:
                T.write(str(i) + '\t')
            T.write('\n')

def Walk(train_items, n_users, n_items, khops):
    coo_rows = []
    coo_cols = []
    coo_data = []
    for uid in train_items.keys():
        coo_rows.extend([uid]*len(train_items[uid]))
        coo_cols.extend(train_items[uid]) 
        coo_data.extend([1.0]*len(train_items[uid]))

    coo_ = sp.coo_matrix((np.array(coo_data), (np.array(coo_rows), np.array(coo_cols))), shape=(n_users, n_items), dtype=np.int16).tocsr()
    coo_T = sp.coo_matrix((np.array(coo_data), (np.array(coo_cols), np.array(coo_rows))), shape=(n_items, n_users), dtype=np.int16).tocsr()

    A_drop = copy.deepcopy(coo_.tocsr()) 
    A_ = copy.deepcopy(coo_T.tocsr()) 
    # 分块
    A_fold = []
    n_fold = 2
    fold_len = n_users // n_fold
    for i_fold in range(n_fold):
        start = i_fold * fold_len
        if i_fold == n_fold -1:
            end = n_users 
        else:
            end = (i_fold + 1) * fold_len
        A_fold.append(A_drop.tocsr()[start:end]) 
    u_id = 0
    candidates = defaultdict(set) 
    for f in range(n_fold):
        if khops == 3:
            A_5 = A_fold[f] * A_ * A_drop           
        elif khops == 5:
            A_5 = A_fold[f] * A_ * A_drop           
        elif khops == 7:
            A_5 = A_fold[f] * A_ * A_drop           
        A_5_array = A_5.tocsr().toarray()
        for i in range(A_5.shape[0]):
            tt1 = time.time()
            index = np.nonzero(np.array(list(A_5_array[i])))  
            intermediate_items = set(index[0]) - set(train_items[u_id])
            intermediate_items = set([i+n_users for i in intermediate_items])
            candidates[u_id].update(intermediate_items)
            u_id += 1
    save_candidates('candidates_zhihu_intermediates.txt', candidates) 


