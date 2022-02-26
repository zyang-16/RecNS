from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)


class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """

    def __init__(self, G, G_pos, G_view, id2idx,
                 placeholders, context_pairs=None, batch_size=100, max_degree=25,
                 n2v_retrain=False, fixed_n2v=False,
                 **kwargs):
        self.G = G
        self.G_pos = G_pos
        self.G_view = G_view
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj_pos, self.deg_pos = self.construct_adj(self.G_pos)
        self.adj_view, self.deg_view = self.construct_adj(self.G_view)
        self.adj, self.deg = self.construct_adj(self.G)
        
        self.save_embed_nodes = list(self.id2idx.values())
        self.save_embed_nodes = np.random.permutation(self.save_embed_nodes)

        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            self.train_edges = self._remove_isolated(self.train_edges)

        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges
        # print("self.train_edges", self.train_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n: self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]


    def _remove_isolated(self, edge_list):
        new_edge_list = []
        for n1, n2 in edge_list:
            new_edge_list.append((n1, n2))
        return new_edge_list


    # def _remove_isolated(self, edge_list):
    #     new_edge_list = []
    #     missing = 0
    #     for n1, n2 in edge_list:
    #         if not n1 in self.nodes or not n2 in self.nodes:
    #             missing += 1
    #             continue
    #         if self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0:
    #             continue
    #         else:
    #             new_edge_list.append((n1, n2))
    #     # print("Unexpected missing:", missing)
    #     return new_edge_list

    def construct_adj(self, graph):
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in graph.nodes():
            neighbors = np.array([self.id2idx[neighbor] for neighbor in graph.neighbors(nodeid)])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['user_ids']: batch1})
        feed_dict.update({self.placeholders['pos_ids']: batch2})
        return feed_dict, batch1, batch2

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = start_idx + self.batch_size
        if end_idx <= len(self.train_edges):
            batch_edges = self.train_edges[start_idx: end_idx]
        else:
            diff = end_idx - len(self.train_edges)
            add_data = self.train_edges[0: diff]
            batch_edges = np.vstack((self.train_edges[start_idx: len(self.train_edges)], add_data))
        return self.batch_feed_dict(batch_edges)


    def test_feed_dict(self, test_edges):
        return self.batch_feed_dict(test_edges)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

    def feed_dict_user(self, start_index, end_index):
        batch1 = list(range(start_index, end_index))
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch1)})
        feed_dict.update({self.placeholders['user_ids']: batch1})
        return feed_dict

    def feed_dict_item(self, start_index, end_index):
        batch2 = list(range(start_index, end_index))
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch2)})
        feed_dict.update({self.placeholders['pos_ids']: batch2})
        return feed_dict
    


