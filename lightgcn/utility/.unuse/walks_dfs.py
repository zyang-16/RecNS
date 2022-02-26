import time
from collections import defaultdict
import numpy as np
from utility.parser import parse_args
from utility.helper import construct_graph
import random
args = parse_args()


class RecNS(object):
    def __init__(self, train_items, n_users):
        self.G = construct_graph(train_items, n_users)
        self.n_users = n_users

    def walk(self, start_node, seen, walks_length):
        stack=[]
        stack.append(start_node)
        seen= set()
        seen.add(start_node)
        traversed = []
        traversed.append(start_node)
        walks = [] 
        hops_num = 0

        while (hops_num < walks_length):  
            hops_num += 1
            if hops_num < args.khop:
                cur=stack.pop() 
                # print("cur")
                nodes=self.G[cur]
                # print("nodes", nodes) 
                neighbors = list(set(nodes) - seen)
                # print("neighbors", neighbors) 
                if len(neighbors) != 0:
                    next_node = np.random.choice(neighbors, 1)[0]
                    # print("next_node", next_node)
                    traversed.append(next_node)
                    stack.append(next_node)
                    seen.add(next_node)
                else:
                    hops_num -= 1
                    print("Truncated.....")
                    # print("cur", cur)
                    remove = traversed.pop()                    
                    print("traversed", traversed)
                    cur = traversed[-1]
                    stack.append(cur)
                    seen.add(cur)
                    # print("stack", stack)   
            else:
                cur=stack.pop()  
                nodes=self.G[cur]
                for w in nodes:
                    if w not in seen:
                        stack.append(w)
                        seen.add(w)
                if cur > self.n_users:
                    walks.append(cur)
                else: 
                    pass
            # print("hops_num", hops_num)
            # print("walks#####", walks)
        # print("walks", walks)
        return walks
    
    def intermediate(self):
        candidate = defaultdict(list)
        for node in self.G.nodes():
            if node < self.n_users:
                # t0 = time.time()
                # print("user_id", node)
                walks = []
                seen = set()
                # print("###############")
                for walk_iter in range(args.num_walks):
                    walks.extend(self.walk(node, seen, args.walk_length))
                # print("time for t0", time.time() - t0) 
                candidate[node].extend(walks)
            else:
                pass         
        return candidate
