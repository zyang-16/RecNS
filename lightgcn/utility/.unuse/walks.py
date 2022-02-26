import time
from collections import defaultdict
import numpy as np
from utility.helper import construct_graph
from utility.parser import parse_args
import random
import queue
args = parse_args()

class RecNS(object):
    def __init__(self, train_items, n_users, n_items):
        self.G = construct_graph(train_items, n_users)
        self.n_users = n_users
        self.n_items = n_items

    def bfs(self, q, seen):
        layer_nodes = set() 
        while not q.empty(): # q is layer queue
            node = q.get()
            neighbors = self.G[node]
            nexts = set(neighbors) - seen
            layer_nodes.update(nexts)
        return layer_nodes 

    def walk(self, start_node, seen, walks_length):
        que = queue.Queue()
        que.put(start_node) 
        seen.add(start_node)
        traversed = []
        walks = [] 
        hops_num = 0 
        while (hops_num < walks_length):   
            hops_num += 1
            if hops_num < args.khop: 
                layer_nodes = self.bfs(que, seen) 
                next_node = np.random.choice(list(set(layer_nodes)), size=10)
                for node in next_node:
                    if node not in seen:
                        que.put(node)
                        seen.add(node)
                if hops_num > 1:
                    cur_pos = next_node[-1]
                    if cur_pos > self.n_users:
                        walks.extend(next_node)
                    else:
                        pass 
            else: 
                layer_nodes = self.bfs(que, seen) 
                next_node = np.random.choice(list(set(layer_nodes)), size=100)
                for node in next_node:
                    if node not in seen:
                        que.put(node)
                        seen.add(node) 
                cur = next_node[-1]
                if cur > self.n_users:
                    walks.extend(next_node)
                else:
                    pass
        return walks
    
    def intermediate(self):
        candidate = defaultdict(list)
        for node in self.G.nodes():
            if node < self.n_users:
                t0 = time.time()
                # print("user_id", node)
                walks = []
                seen = set()
                # print("###############")
                for walk_iter in range(args.num_walks):
                    walks.extend(self.walk(node, seen, args.walk_length))
                print("time for t0", time.time() - t0) 
                candidate[node].extend(walks)
            else:
                pass         
        return candidate

