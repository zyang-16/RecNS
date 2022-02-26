import tensorflow as tf
from collections import namedtuple
import numpy as np
import math
from graphsage.layers import Layer
import graphsage.layers as layers
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method
    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)


# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',  # name of the layer (to get feature embedding etc.)
                       'num_samples',  # sample num for each layer
                       'output_dim'  # the output (i.e., hidden) dimension
                       ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, feature_user, feature_item, adj, adj_pos, adj_view, degrees,
                 layer_infos, concat=True, aggregator_type="mean",
                 model_size="small", identity_dim=0, args=None,
                 **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["user_ids"]
        self.inputs2 = placeholders["pos_ids"]
        self.neg_samples = placeholders["neg_ids"]
        self.neg_samples_exposed = placeholders["neg_ids_exposed"]
        self.neg_size = placeholders["neg_size"]
        self.neg_size_exposed = placeholders["neg_size_exposed"]
        self.beta = placeholders["beta"]
        self.model_size = model_size
        self.adj_info = adj
        self.adj_info_pos = adj_pos
        self.adj_info_view = adj_view
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None

        # user feature processing
        if feature_user is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features_user = None
        else:
            self.feature_u = tf.Variable(tf.constant(feature_user, dtype=tf.float32), trainable=False)
            self.features_user = tf.layers.dense(self.feature_u, args.identity_dim, activation=tf.nn.tanh, use_bias=True)

        # item feature processing
        if feature_item is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features_item = None
        else:
            feature_item = np.vstack([feature_item, np.zeros((feature_item.shape[1], ))])
            self.feature_i = tf.Variable(tf.constant(feature_item, dtype=tf.float32), trainable=False)
            self.features_item = tf.layers.dense(self.feature_i, args.identity_dim, activation=tf.nn.tanh, use_bias=True)

        if feature_user is None and feature_item is None:
            self.feature = None
            self.features = self.embeds
        else:
            self.feature = tf.concat([self.features_user, self.features_item], axis=0)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.feature], axis=1)
        
        self.degrees = degrees
        self.concat = concat
        self.margin = 1.0

        self.dims = [(0 if self.feature is None else args.identity_dim) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))]) #[100, 256, 256]
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.args = args

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)

        self.build()

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
                  aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None # aggregator is None, new_agg=True, else new_agg=False
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1], act=lambda x: x,
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult * dims[layer], dims[layer + 1],
                                                     dropout=self.placeholders['dropout'],
                                                     name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              dim_mult * dims[layer]]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_dims))) # _call of aggregator
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators
    
    def neighborsampler(self, inputs, adj_info):
        """
        Uniformly samples neighbors.
        Assumes that adj lists are padded with random re-sampling
        """
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists

    def sample(self, inputs, layer_infos, adj_info, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            # sampler = layer_infos[t].neigh_sampler
            node = self.neighborsampler((samples[k], layer_infos[t].num_samples), adj_info)  # _call() function of UniformNeighborSampler
            samples.append(tf.reshape(node, [support_size * batch_size, ]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def _build(self):
        # perform "convolution"
        # pos convolution for node embedding
        samples1_pos, support_sizes1_pos = self.sample(self.inputs1, self.layer_infos, self.adj_info_pos) # support_sizes = [1, 10, 250]
        samples2_pos, support_sizes2_pos = self.sample(self.inputs2, self.layer_infos, self.adj_info_pos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos] # [25, 10]
        self.outputs1, self.aggregators_pos = self.aggregate(samples1_pos, [self.features], self.dims, num_samples,
                                                         support_sizes1_pos, concat=self.concat, model_size=self.model_size)
        self.outputs2, _ = self.aggregate(samples2_pos, [self.features], self.dims, num_samples,
                                          support_sizes2_pos, aggregators=self.aggregators_pos, concat=self.concat,
                                          model_size=self.model_size)
        
        neg_samples_pos, neg_support_sizes_pos = self.sample(self.neg_samples, self.layer_infos, self.adj_info_pos, self.neg_size)

        self.neg_outputs_pos, _ = self.aggregate(neg_samples_pos, [self.features], self.dims, num_samples,
                                             neg_support_sizes_pos, batch_size=self.neg_size,
                                             aggregators=self.aggregators_pos,
                                             concat=self.concat, model_size=self.model_size)
        
        neg_samples_view, neg_support_sizes_view = self.sample(self.neg_samples_exposed, self.layer_infos, self.adj_info_pos, self.neg_size_exposed)

        self.neg_outputs_view, _ = self.aggregate(neg_samples_view, [self.features], self.dims, num_samples,
                                             neg_support_sizes_view, batch_size=self.neg_size_exposed,
                                             aggregators=self.aggregators_pos,
                                             concat=self.concat, model_size=self.model_size)
        

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, dim=1)  # bacth*dim = N*K
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, dim=1)  # batch*dim = N*K
        self.neg_outputs_pos = tf.nn.l2_normalize(self.neg_outputs_pos, dim=1) # (batch*neg_size) * dim= (N*W)*K
        self.neg_outputs_view = tf.nn.l2_normalize(self.neg_outputs_view, dim=1) # (batch*neg_size) * dim= (N*W)*K

        # merge embedding
        self.ids = tf.constant(np.arange(0, self.args.batch_size*self.args.K_negs, self.args.K_negs))
        self.neg_i_g_embeddings_pos_one = tf.nn.embedding_lookup(self.neg_outputs_pos, self.ids)
        self.neg_i_g_embeddings_merge = (1-1.0/self.args.K_negs) * self.neg_i_g_embeddings_pos_one + (1.0/self.args.K_negs) * self.neg_outputs_view
        self.neg_outputs_exposed_copy = tf.reshape(tf.tile(self.neg_i_g_embeddings_merge, [1, self.args.K_negs]), (-1, tf.shape(self.neg_outputs_view)[-1]))
        self.neg_concat = (1-1.0/self.args.K_negs) * self.neg_outputs_pos + (1.0/self.args.K_negs) * self.neg_outputs_exposed_copy
        self.neg_outputs = tf.nn.l2_normalize(self.neg_concat, dim=1)
        
        for aggregator in self.aggregators_pos:
            for var in aggregator.vars.values():
                tf.add_to_collection("params", var)
        

    def build(self):
        self._build() 

        # TF graph management
        self._loss()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        self.loss = self.loss / tf.cast(self.args.K_negs, tf.float32)
        tf.summary.scalar('loss', self.loss)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars] 
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars) 
        
        '''
        negative sampling rating 
        '''
        # positive-assisted sampling
        alpha = tf.random_uniform((self.batch_size, self.args.K_Candidates), minval=0, maxval=1.0)
        self.all_rating_user = tf.squeeze(tf.matmul(tf.reshape(self.neg_outputs_pos, (self.batch_size, -1, self.dims[-1]*2)), tf.expand_dims(self.outputs1, -1)))
        self.all_rating_item = tf.squeeze(tf.matmul(tf.reshape(self.neg_outputs_pos, (self.batch_size, -1, self.dims[-1]*2)), tf.expand_dims(self.outputs2, -1)))
        self.all_rating = tf.sigmoid(alpha * self.all_rating_user + (1-alpha) * self.all_rating_item)
        self.negs_id = tf.multinomial(self.all_rating_user, self.args.K_negs)
        # exposure-augmented sampling
        self.all_rating_user_exposed = tf.squeeze(tf.matmul(tf.reshape(self.neg_outputs_view, (self.batch_size, -1, self.dims[-1]*2)), tf.expand_dims(self.outputs1, -1)))
        self.prob_beta = tf.sigmoid(self.all_rating_user_exposed * self.beta)
        self.negs_id_exposed = tf.argmax(self.prob_beta, 1)

    # loss functions 
    def _loss(self):
        self.params = tf.get_collection("params")
        for var in self.params:
            self.loss += self.args.weight_decay * tf.nn.l2_loss(var) 

        self.outputs1_copy = tf.reshape(tf.tile(self.outputs1,[1,self.args.K_negs]), (-1, self.dims[1]*2))
        self.outputs2_copy = tf.reshape(tf.tile(self.outputs2,[1,self.args.K_negs]), (-1, self.dims[1]*2))
        self.pos_aff = tf.sigmoid(tf.reduce_sum(self.outputs1_copy * self.outputs2_copy, axis=1), name="pos_aff")
        self.neg_aff = tf.sigmoid(tf.reduce_sum(self.outputs1_copy * self.neg_outputs, axis=1), name="neg_aff") 
        hinge_loss = tf.reduce_sum(tf.nn.relu(tf.subtract(self.neg_aff, self.pos_aff - 0.1)))

        self.loss += hinge_loss



