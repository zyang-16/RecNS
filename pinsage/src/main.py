import argparse
from samplers.ns import *
from samplers.walks import Walk
from graphsage.model import SAGEInfo, SampleAndAggregate
from graphsage.minibatch import EdgeMinibatchIterator
import time
import os
import numpy as np
import tensorflow as tf
from evaluate import recommend


def parse_args(): 
    parser = argparse.ArgumentParser(description="run baselines to generate negative items")
    parser.add_argument('--input', type=str, default='../data/zhihu/',
                        help='input dataset path.')
    parser.add_argument('--strategy', type=str, default='RecNS',
                        help='negative sampling strategy name.')
    parser.add_argument('--model', type=str, default='graphsage_mean',
                        help='model name.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate.') 
    parser.add_argument('--model_size', type=str, default="small",
                        help='big or small.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs.')
    parser.add_argument('--validate_iter', type=int, default=50,
                        help='how often to run a validation minibatch.')
    parser.add_argument('--print_step', type=int, default=500,
                        help='how often to print training info.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='number of batch size.')
    parser.add_argument('--save_size', type=int, default=1024,
                        help='number of saved_model batch size.')                   
    parser.add_argument('--dim', type=int, default=256,
                        help='number of dimension.') 
    parser.add_argument('--samples_1', type=int, default=25,
                        help='number of samples in  layer 1.')
    parser.add_argument('--samples_2', type=int, default=10,
                        help='number of samples in layer 2.')
    parser.add_argument('--dim_1', type=int, default=128,
                        help='size of output dim.')
    parser.add_argument('--dim_2', type=int, default=128,
                        help='size of output dim.')
    parser.add_argument('--att_dim', type=int, default=128,
                        help='Number of attention dimensions.')
    parser.add_argument('--max_degree', type=int, default=500,
                        help='maximum node degree.')
    parser.add_argument('--user_num', type=int, default=0,
                        help='number of users')
    parser.add_argument('--item_num', type=int, default=0,
                        help='number of items.')
    parser.add_argument('--K_negs', type=int, default=15,
                        help='number of negative items for each pair.')
    parser.add_argument('--K_Candidates', type=int, default=20,
                        help='candidate number.') 
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature') 
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight for l2 loss on embedding matrix.')
    parser.add_argument('--identity_dim', type=int, default=50,
                        help='set to positive value to use identity embedding features of that dimension.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate.') 
    parser.add_argument('--khops', type=int, default=3,
                        help='--negative_k_hop_sampling')  
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping.')
    parser.add_argument('--save_dir', type=str, default="./embeddings/",
                        help='save embeddings path.') 
    parser.set_defaults(undirected=False) 
    return parser.parse_args() 


def save_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    # save user embedding
    start_index = 0
    while start_index < User_Num:
        end_index = start_index + size
        if end_index < User_Num:
            feed_dict_user = minibatch_iter.feed_dict_user(start_index, end_index)
            embed = sess.run(model.outputs1, feed_dict=feed_dict_user)
        else:
            feed_dict_user = minibatch_iter.feed_dict_user(start_index, User_Num)
            embed = sess.run(model.outputs1, feed_dict=feed_dict_user)
        if start_index == 0:
            user_embed = embed
        else:
            user_embed = np.vstack((user_embed, embed)) 
        start_index = end_index
    print("user_embed.shape", user_embed.shape)

    # save item embedding  
    item_embed = list()  
    start_index = User_Num
    while start_index < User_Num + Item_Num:
        end_index = start_index + size
        if end_index < User_Num + Item_Num:
            feed_dict_item = minibatch_iter.feed_dict_item(start_index, end_index)
            embed = sess.run(model.outputs2, feed_dict=feed_dict_item)
        else:
            feed_dict_item = minibatch_iter.feed_dict_item(start_index, User_Num+Item_Num)
            embed = sess.run(model.outputs2, feed_dict=feed_dict_item)
        if start_index == User_Num:
            item_embed = embed
        else:
            item_embed = np.vstack((item_embed, embed)) 
        start_index = end_index
    print("item_embed.shape", item_embed.shape)
    name1 = 'embedding_u'
    name2 = 'embedding_v'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    embeddings_u = np.vstack(user_embed)
    np.save(out_dir + name1 + mod + ".npy", embeddings_u)

    embeddings_v = np.vstack(item_embed)
    np.save(out_dir + name2 + mod + ".npy", embeddings_v)
    


def construct_placeholders():
    # Define placeholders
    placeholders = {
        'user_ids': tf.placeholder(tf.int32, shape=(None), name='user_ids'),
        'pos_ids': tf.placeholder(tf.int32, shape=(None), name='pos_ids'),
        'neg_ids': tf.placeholder(tf.int32, shape=(None), name='neg_ids'),
        'neg_ids_exposed': tf.placeholder(tf.int32, shape=(None), name='neg_ids_exposed'),
        'neg_size': tf.placeholder(tf.int32, shape=(None), name='neg_size'),
        'neg_size_exposed': tf.placeholder(tf.int32, shape=(None), name='neg_size_exposed'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        'beta': tf.placeholder(tf.float32, shape=(None), name='beta'),
        'probs': tf.placeholder(tf.float32, shape=(None), name='probs'),
    }
    return placeholders


def train(train_data, feature_dict_user,feature_dict_item, args):
    # generate negative items for each positive user-item pairs.
    context_pairs = train_data_by_type['Pos']  # positive user-item pairs

    # build model & create tensorflow graph
    graph = tf.Graph()
    G = construct_graph(train_data)  # train data (positive + negative)
    G_pos = construct_graph(train_data_by_type['Pos'])  
    G_view = construct_graph(train_data_by_type['Neg'])  

    id_map = dict(zip([node for node in range(User_Num + Item_Num)], [node for node in range(User_Num + Item_Num)]))
    with graph.as_default():
        placeholders = construct_placeholders()
        minibatch = EdgeMinibatchIterator(G, G_pos, G_view, id_map, placeholders,
                                          batch_size=args.batch_size, max_degree=args.max_degree,
                                          context_pairs=context_pairs)

        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        adj_info_ph_pos = tf.placeholder(tf.int32, shape=minibatch.adj_pos.shape)
        adj_info_pos = tf.Variable(adj_info_ph_pos, trainable=False, name="adj_info_pos")

        adj_info_ph_view = tf.placeholder(tf.int32, shape=minibatch.adj_view.shape)
        adj_info_view = tf.Variable(adj_info_ph_view, trainable=False, name="adj_info_view")
        
        # embedding-based model or discriminator
        if args.model == 'graphsage_mean':
            layer_infos = [SAGEInfo("node", args.samples_1, args.dim_1),
                           SAGEInfo("node", args.samples_2, args.dim_2)] 
            # layer_infos = [SAGEInfo("node", args.samples_1, args.dim_1)]
            print("layer_infos", layer_infos)
            model = SampleAndAggregate(placeholders, feature_user, feature_item, adj_info, adj_info_pos, adj_info_view, minibatch.deg,
                                       layer_infos=layer_infos, model_size=args.model_size,
                                       identity_dim=args.identity_dim, args=args, logging=True)

        elif args.model == 'gcn':
            layer_infos = [SAGEInfo("node", args.samples_1, 2*args.dim_1),
                           SAGEInfo("node", args.samples_2, 2*args.dim_2)]
            model = SampleAndAggregate(placeholders, feature_user, feature_item, adj_info, adj_info_pos, adj_info_view, minibatch.deg,
                                       layer_infos=layer_infos, aggregator_type="gcn",
                                       model_size=args.model_size, identity_dim=args.identity_dim,
                                       concat=False, args=args, logging=True)

        elif args.model == 'graphsage_seq':
            layer_infos = [SAGEInfo("node", args.samples_1, args.dim_1),
                           SAGEInfo("node", args.samples_2, args.dim_2)]
            model = SampleAndAggregate(placeholders, feature_user, feature_item, adj_info, adj_info_pos, adj_info_view, minibatch.deg,
                                       layer_infos=layer_infos, aggregator_type="seq",
                                       model_size=args.model_size, identity_dim=args.identity_dim,
                                       args=args, logging=True)

        elif args.model == 'graphsage_maxpool':
            layer_infos = [SAGEInfo("node", args.samples_1, args.dim_1),
                           SAGEInfo("node", args.samples_2, args.dim_2)]
            model = SampleAndAggregate(placeholders, feature_user, feature_item, adj_info, adj_info_pos, adj_info_view, minibatch.deg,
                                       layer_infos=layer_infos, aggregator_type="maxpool",
                                       model_size=args.model_size, identity_dim=args.identity_dim,
                                       args=args, logging=True)

        elif args.model == 'graphsage_meanpool':
            layer_infos = [SAGEInfo("node", args.samples_1, args.dim_1),
                           SAGEInfo("node", args.samples_2, args.dim_2)]
            model = SampleAndAggregate(placeholders, feature_user, feature_item, adj_info, adj_info_pos, adj_info_ph_view, minibatch.deg,
                                       layer_infos=layer_infos, aggregator_type="meanpoo",
                                       model_size=args.model_size, identity_dim=args.identity,
                                       args=args, logging=True)

        else:
            raise Exception('Error: model name unrecognized')
        # initialize
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

    # optimization
    print("Optimizing.........")
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config, graph=graph) as sess:
        sess.run(init, feed_dict={adj_info_ph: minibatch.adj, adj_info_ph_pos: minibatch.adj_pos, adj_info_ph_view: minibatch.adj_view})
        summary_write = tf.summary.FileWriter("./logs/", sess.graph)
        print("Training...")
        total_steps = 0
        # best_auc = 0
        best_recall = 0 
        patience = 0
        iter = 0 
        
        t_candidates = time.time()
        Walk(train_items, User_Num, Item_Num, args.khops)
        intermediates = load_candidates('candidates_zhihu_intermediates.txt')
        print("loading candidates......", time.time() - t_candidates)
        
        for epoch in range(args.epochs):
            minibatch.shuffle()
            print('Epoch: %04d' % (epoch + 1)) 
            it = 0
            while not minibatch.end():
                feed_dict, batch1, batch2 = minibatch.next_minibatch_feed_dict() 
                feed_dict.update({placeholders['dropout']: args.dropout})                
                # negative sampling strategies to generate neagtive items from unobserved item sets
                negative_items, negative_items_exposed = negative_sampling(sess, model, intermediates, train_rec, exposed_items_dict, batch1, batch2, args) # batch * K 
                feed_dict.update({placeholders['neg_ids']: negative_items})
                feed_dict.update({placeholders['neg_ids_exposed']: negative_items_exposed})
                feed_dict.update({placeholders['neg_size']: len(negative_items)})
                feed_dict.update({placeholders['neg_size_exposed']: len(negative_items_exposed)})

                train_time = time.time()                
                outs = sess.run([merged, model.opt_op, model.loss], feed_dict=feed_dict)
                train_cost = outs[2]

                if total_steps % args.print_step == 0:
                    summary_write.add_summary(outs[0], total_steps)
                    print("Iter:", '%04d' % iter,
                        "train_loss=", "{:.5f}".format(train_cost),
                        "time=", "{:.5f}".format(time.time() - train_time))
                
                iter += 1
                it += 1
                total_steps += 1
            
            if epoch % 10 == 0:
                # save model (node embeddings)
                # save user embedding
                saved_model = dict()
                start_index = 0
                while start_index < User_Num:
                    end_index = start_index + args.save_size
                    if end_index < User_Num:
                        feed_dict_user = minibatch.feed_dict_user(start_index, end_index)
                        embed = sess.run(model.outputs1, feed_dict=feed_dict_user)
                    else:
                        feed_dict_user = minibatch.feed_dict_user(start_index, User_Num)
                        embed = sess.run(model.outputs1, feed_dict=feed_dict_user)
                    if start_index == 0:
                        user_embed = embed
                    else:
                        user_embed = np.vstack((user_embed, embed)) 
                    start_index = end_index
                print("user_embed.shape", user_embed.shape)

                # save item embedding  
                item_embed = list()  
                start_index = User_Num
                while start_index < User_Num + Item_Num:
                    end_index = start_index + args.save_size
                    if end_index < User_Num + Item_Num:
                        feed_dict_item = minibatch.feed_dict_item(start_index, end_index)
                        embed = sess.run(model.outputs2, feed_dict=feed_dict_item)
                    else:
                        feed_dict_item = minibatch.feed_dict_item(start_index, User_Num+Item_Num)
                        embed = sess.run(model.outputs2, feed_dict=feed_dict_item)
                    if start_index == User_Num:
                        item_embed = embed
                    else:
                        item_embed = np.vstack((item_embed, embed)) 
                    start_index = end_index
                print("item_embed.shape", item_embed.shape)
                saved_model['user'] = np.array(user_embed) 
                saved_model['item'] = np.array(item_embed)

                # validation for each epoch with recommendation
                valid_recall, valid_ndcg, valid_hr = recommend(saved_model, valid_rec, train_rec, args, metrics=['recall', 'ndcg', 'hit_rate'], k_list=[20, 50])
                k_list = [20, 50]
                for index in list(range(len(k_list))):
                    print("metric@", "{:d}".format(k_list[index]))
                    print("Epoch:", '%04d' % (epoch + 1),
                            "val_recall=", "{:.5f}".format(valid_recall[index]),
                            "val_ndcg=", "{:.5f}".format(valid_ndcg[index]),
                            "val_hr=", "{:.5f}".format(valid_hr[index]),
                            "time=", "{:.5f}".format(time.time() - train_time))
                curr_recall = valid_recall[0]
                if curr_recall > best_recall:
                    best_recall = curr_recall
                    patience = 0
                else:
                    patience += 1
                    if patience > args.patience:
                        print("Early Stopping...")
                        break
        print("Optimization Finished!")
        # save model embeddings for downstream task
        save_embeddings(sess, model, minibatch, args.save_size, args.save_dir)
        
    # test for recommendation
    saved_model = load_test_embedding(args)
    test_recall, test_ndcg, test_hr = recommend(saved_model, test_rec, train_rec, args, metrics=['recall', 'ndcg', 'Hit_rate'], k_list=[20, 50])

    return test_recall, test_ndcg, test_hr 


if __name__ == "__main__":
    args = parse_args()
    filepath = args.input
    print("lr", args.learning_rate)
    print("weight_decay", args.weight_decay)
    # load node features
    if os.path.exists(filepath + 'user_feature.txt'):
        feature_user, User_Num = load_features(filepath + 'user_feature.txt')
        feature_item, Item_Num = load_features(filepath + 'item_feature.txt')
    else:
        feature_user = None
        feature_item = None
        User_Num = args.user_num
        Item_Num = args.item_num
    print("User_Num", User_Num)
    print("Item_Num", Item_Num)
    # load train, test, valid data 
    train_data_by_type, train_data = load_train_data(filepath + 'train.txt')
    print("train_data", len(train_data))
    print("train_data_positive", len(train_data_by_type['Pos']))
    print("train_data_exposure", len(train_data_by_type['Neg']))
    valid_data, valid_rec = load_test_data(filepath + 'valid.txt')
    test_data, test_rec = load_test_data(filepath + 'test.txt')  
    train_rec = load_test_rec(train_data_by_type['Pos'])
    train_items = load_train_items(train_data_by_type['Pos'], User_Num)
    mask = load_test_rec(train_data_by_type['Pos'])
    exposed_items_dict = load_test_rec(train_data_by_type['Neg'])
    test_recall, test_ndcg, test_hr = train(train_data, feature_user, feature_item, args)
    k_list = [20, 50]
    for index in list(range(len(k_list))):
        print("metric@", "{:d}".format(k_list[index]))
        print("test_recall=", "{:.5f}".format(test_recall[index]),
            "test_ndcg=", "{:.5f}".format(test_ndcg[index]),
            "test_hr=", "{:.5f}".format(test_hr[index]))        
