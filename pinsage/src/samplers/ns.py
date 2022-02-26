import numpy as np
from load_data import *
import time
from scipy.stats import norm
import random

def ns_recns_o_1(sess, model, candidates, item_list, train_items, exposed_items_dict, user_batch, item_batch, K_Candidates, K_negs, args):
    batch_neg = []
    batch_neg_exposed = []
    neg_items = []
    neg_items_exposed = []
    beta = np.ones((len(user_batch), K_Candidates))
    for i, user in enumerate(user_batch):
        tt = time.time()
        negatives = random.sample(candidates[user], K_Candidates) 
        batch_neg.extend(negatives)
        if user in exposed_items_dict.keys():
            if len(exposed_items_dict[user])<K_Candidates:
                exposed_items = np.random.choice(exposed_items_dict[user], K_Candidates)
            else:
                exposed_items = random.sample(exposed_items_dict[user], K_Candidates)
            exposed_items_in_candidates = [neg for neg in negatives if neg in exposed_items]
            beta_exposed = len(exposed_items_in_candidates)
            if beta_exposed >= 2:
                for j, neg in enumerate(exposed_items):
                    if neg in negatives:
                        beta[i][j] = beta_exposed
            batch_neg_exposed.extend(exposed_items)
        else:
            exposed_items = negatives
            batch_neg_exposed.extend(exposed_items)
    negs_id, negs_id_exposed, prob_beta = sess.run([model.negs_id, model.negs_id_exposed, model.prob_beta], feed_dict={model.inputs1:user_batch, model.inputs2:item_batch, model.neg_samples:batch_neg, model.batch_size: len(user_batch), model.neg_size:len(batch_neg), model.neg_samples_exposed: batch_neg_exposed, model.neg_size_exposed:len(batch_neg_exposed), model.beta: beta})
    batchs = np.array(batch_neg).reshape((len(user_batch), -1)) 
    neg_items = np.squeeze(np.reshape([[negs[id]for id in index] for negs, index in zip(batchs, negs_id)], (1,-1))) 
    batchs_exposed = np.array(batch_neg_exposed).reshape((len(user_batch), -1)) 
    neg_items_exposed = np.squeeze(np.reshape([[negs[id]for id in index] for negs, index in zip(batchs_exposed, np.array(negs_id_exposed).reshape(-1,1))], (1,-1))) 
    return neg_items, neg_items_exposed 


def negative_sampling(sess, model, candidates, train_items, exposed_items_dict, user_batch, item_batch, args):
    item_list = list(range(args.user_num, args.user_num+args.item_num))
    negative_items, negative_items_exposed = ns_recns_o_1(sess, model, candidates, item_list, train_items, exposed_items_dict, user_batch, item_batch, args.K_Candidates, args.K_negs, args)
    return negative_items, negative_items_exposed



