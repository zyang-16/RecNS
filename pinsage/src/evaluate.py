import numpy as np
import faiss

def compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len([v for v in targets if v in pred]) 
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def dcg_at_k(hits_list):
    return np.sum(hits_list / np.log2(np.arange(2,len(hits_list)+2)))


def compute_ndcg(items, recommended_items, k):
    user_hits = []
    recommended_tops = recommended_items[:k]
    for i in recommended_tops:
        pred = 1 if i in items else 0
        user_hits.append(pred)
    user_hits = np.array(user_hits, dtype=np.float32)
    if len(items) >= k:
        ideal_rels = np.ones(k)
    else:
        ideal_rels = np.pad(np.ones(len(items)), (0, k-len(items)), 'constant')
    ndcg = dcg_at_k(user_hits) / dcg_at_k(ideal_rels)
    return ndcg


def compute_hr(items, recommended_items, k):
    pred = recommended_items[:k]
    num_hit = len([v for v in items if v in pred]) 
    if num_hit > 0:
        return 1.0
    else:
        return 0.0


def recommend(model, test_rec, train_rec, args, metrics, k_list=None):
    '''
    recall@k, precision@k, ndcg@k, map@k
    '''
    recalls = []
    ndcgs = []
    hrs = []

    rec_list = [list() for _ in range(len(k_list))]
    ndcg_list = [list() for _ in range(len(k_list))]
    hr_list = [list() for _ in range(len(k_list))]
    User_Num = args.user_num

    item_embeds = model['item'] # item_num * dim

    # faiss
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.dim, flat_config)
        gpu_index.add(item_embeds)
    except Exception as e:
        return {}
    # print("gpu_index", gpu_index)

    for user, items in test_rec.items():
        user_embeds = np.expand_dims(model['user'][user], axis=0)
        D, I = gpu_index.search(user_embeds, 200) 
        recommended_items_200 = I[0] + User_Num
        rated = set(train_rec[user])
        recommended_items_unobserved = [p for p in recommended_items_200 if p not in rated]
        for i, k in enumerate(k_list):  
            recommended_items = recommended_items_unobserved[:k]      
            prec, recall = compute_precision_recall(items, recommended_items, k)
            rec_list[i].append(recall)
            ndcg_list[i].append(compute_ndcg(items, recommended_items, k))
            hr_list[i].append(compute_hr(items, recommended_items, k))
    
    rec_list = [np.array(i) for i in rec_list]
    ndcg_list = [np.array(i) for i in ndcg_list]
    hr_list = [np.array(i) for i in hr_list]

    for i in list(range(len(k_list))):
        recalls.append(np.mean(rec_list[i]))
        ndcgs.append(np.mean(ndcg_list[i]))
        hrs.append(np.mean(hr_list[i]))
    return recalls, ndcgs, hrs

