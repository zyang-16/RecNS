import networkx as nx
from collections import defaultdict
import numpy as np

def construct_graph(edges):
    print("Construct the graph for training")
    G = nx.Graph()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        G.add_edge(x, y)
    return G


def count_num(neighbors_list):
    nums = defaultdict(int) 
    for i in neighbors_list:
        nums[i] += 1
    sum_num = sum(nums.values())
    # print("sum_num", sum_num)
    for key in nums:
        nums[key] = nums[key] / sum_num
    return nums


# load train edges
def load_train_data(filename):
    print("Loading train data......")
    edge_data_by_type = defaultdict(list)
    all_edges = list()
    conversion = lambda n : int(n)
    with open(filename, 'r') as f:
        for line in f:
            user, item, type = line.strip().split('\t')
            edge_data_by_type[type].append((int(user), int(item)))
            # edge_data_by_type[type].append(map(conversion, [user, item]))
            all_edges.append((int(user), int(item)))
    all_edges = list(set(all_edges))
    return edge_data_by_type, all_edges


# load test data
def load_test_data(filename):
    print("Loading test/valid data......")
    edges = list()
    test_rec = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            user, item = line.strip().split('\t')
            edges.append((int(user), int(item)))
            test_rec[int(user)].append(int(item))
    return edges, test_rec


def load_features(filepath):
    if filepath is not None:
        features = list()
        num = list()
        with open(filepath, 'r') as f:
            for line in f:
                dd = line.strip().split(' ')
                features.append(dd[1:])
                num.append(dd[0])
        features = np.array(features, dtype=np.int)
        number = len(num)
    else:
        features = None
        number = None
    return features, number

def load_test_rec(test_data):
    test_rec = defaultdict(list)
    for edge in test_data:
        u = edge[0]
        v = edge[1]
        test_rec[u].append(v)
    return test_rec

def load_train_items(test_data, n_users):
    test_rec = defaultdict(list)
    for edge in test_data:
        u = edge[0]
        v = edge[1] - n_users
        test_rec[u].append(v)
    return test_rec

def load_test_embedding(args):
    saved_model = dict()
    embeds_u = np.load(args.save_dir + "/embedding_u.npy")
    embeds_v = np.load(args.save_dir + "/embedding_v.npy")
    saved_model['user'] = np.array(embeds_u) 
    saved_model['item'] = np.array(embeds_v)
    return saved_model


def get_popularity(train_rec):
    popularity_dict = defaultdict(set)
    for user in train_rec.keys():
        for item in train_rec[user]:
            popularity_dict[item].add(user)
    popularity_dict = {key: len(val) for key, val in popularity_dict.items()}
    return popularity_dict    


def calcItemPopScore(train_data, num_items, num_users):
    item_counts = get_popularity(train_data)
    items_pop_score = np.zeros([num_items, 1], dtype=np.float32)
    total_count = np.sum(list(item_counts.values()))
    for iid in list(range(num_users, num_users+num_items)):
        if iid in item_counts:
            items_pop_score[iid-num_users] = float(item_counts[iid]) / total_count
        else:
            items_pop_score[iid-num_users] = 0.0
    return np.squeeze(items_pop_score) 


def get_length(walks):
    length = 0
    for key in walks.keys():
        length += len(walks[key])
    return length

def calcItemDict(true_edges):
    item_pop = list()
    node_deg = dict()
    dd = defaultdict(list)
    for edge in true_edges:
        dd[int(edge[1])].append(int(edge[0]))
    for key in dd.keys():
        item_pop.append(1)
    deg_sum = np.sum(item_pop)
    for key in dd.keys():
        node_deg[key] = 1/deg_sum
    return node_deg, dd


def get_user_batch(user_list, args, batch_num):
    start_idx = batch_num * args.save_size
    batch_num += 1
    end_idx = start_idx + args.save_size
    if end_idx <= len(user_list):
        user_batch = user_list[start_idx: end_idx]
    else:
        user_batch = user_list[start_idx: len(user_list)]
    return user_batch, batch_num



from collections import defaultdict
import random
def load_candidates(filename):
    candidates = defaultdict(set)
    with open(filename, 'r') as f:
        for line in f:
            ddd = line.strip().split('\t')
            for sample in ddd[1:]:
                candidates[int(ddd[0])].add(int(sample))
    candidates_ok = defaultdict(list)
    for user in candidates.keys():
        candidates_ok[user].extend(list(candidates[user]))
    return candidates_ok


