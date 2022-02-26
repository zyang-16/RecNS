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
        self.dis_nodes = list(set(list(range(self.n_users, self.n_users+self.n_items))) - set(self.G.nodes()))
        self.dis_items = []
        for node in self.dis_nodes:
            if node < self.n_users:
                pass
            else:
                self.dis_items.append(node)


    

