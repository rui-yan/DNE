import numpy as np
import tensorflow as tf
from .le import LE
from .svd import SVD
import sys
sys.path.append("..")
from utils.utils_graph import preprocess_nxgraph
from utils.sampler import UnsupervisedSampler, LinkGenerator

class MLP_encoder(tf.keras.Model):
    def __init__(self, hidden_dim=128):
        super(MLP_encoder, self).__init__()
        self.feat_enc = tf.keras.layers.Dense(hidden_dim, use_bias=True, activation="relu") #
        # self.layer_norm = tf.keras.layers.LayerNormalization()
        
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(hidden_dim, activation='gelu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
        ])
    
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        pos_emb = inputs[0]
        embeddings = [pos_emb]
        
        # Check if feat is not None and feat_enc layer exists
        if len(inputs) > 1 and inputs[1] is not None:
            feat = inputs[1]
            feat_emb = self.feat_enc(feat)  # Feature encoding
            embeddings.append(feat_emb)
        
        # Concatenate embeddings and project if necessary
        combined_emb = tf.concat(embeddings, axis=-1) if len(embeddings) > 1 else embeddings[0]
        # normalized_emb = self.layer_norm(combined_emb)
        dense_output = self.dense(combined_emb)  # Pass the combined embedding through the dense layers
        
        return dense_output

def build_model(hidden_dim=128, pos_embed_size=256, feat_size=None, metric='l1', out_activation='linear'):
    encoder = MLP_encoder(hidden_dim)
    x_pos_inp_1 = tf.keras.Input(shape=(pos_embed_size,))
    x_pos_inp_2 = tf.keras.Input(shape=(pos_embed_size,))
    
    if feat_size is not None:
        x_feat_inp_1 = tf.keras.Input(shape=(feat_size,))
        x_feat_inp_2 = tf.keras.Input(shape=(feat_size,))
        x_out_1 = encoder([x_pos_inp_1, x_feat_inp_1])
        x_out_2 = encoder([x_pos_inp_2, x_feat_inp_2])
        x_inp = [x_pos_inp_1, x_feat_inp_1, x_pos_inp_2, x_feat_inp_2]
    else:
        x_out_1 = encoder([x_pos_inp_1])
        x_out_2 = encoder([x_pos_inp_2])
        x_inp = [x_pos_inp_1, x_pos_inp_2]
    
    x_out = [x_out_1, x_out_2]
    
    similarity_metrics = {
        'dot': lambda x: tf.keras.layers.Dot(axes=-1, normalize=True)(x),
        'cosine': lambda x: tf.keras.layers.Dot(axes=-1, normalize=True)(x),
        'l1': lambda x: tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))(x),
        'l2': lambda x: tf.keras.layers.Lambda(lambda x: tf.square(x[0] - x[1]))(x),
        'hadamard': lambda x: tf.keras.layers.Lambda(lambda x: tf.multiply(x[0], x[1]))(x),
        'concat': lambda x: tf.keras.layers.Concatenate()([x[0], x[1]]),
        'avg': lambda x: tf.keras.layers.Average()([x[0], x[1]]),
    }
    
    similarity_layer = similarity_metrics[metric](x_out)
    if metric == 'dot':
        prediction = tf.keras.layers.Reshape((1,))(similarity_layer)
    else:
        prediction = tf.keras.layers.Dense(1, activation=out_activation)(similarity_layer)
    
    model = tf.keras.Model(inputs=x_inp, outputs=prediction)
    
    if feat_size is not None:
        embed_model = tf.keras.Model(inputs=[x_pos_inp_1, x_feat_inp_1], outputs=x_out_1)
    else:
        embed_model = tf.keras.Model(inputs=[x_pos_inp_1], outputs=x_out_1)
    
    return model, embed_model


class DNE:
    def __init__(self, graph, hidden_dim=128, pos_embed_size=256, feat_size=None, 
                 pos_embed='LE', node_attributes=None, metric='l1', out_activation='sigmoid', 
                 hidden_dropout_prob=0.1, walks=None):
        super().__init__()
        
        self.graph = graph
        self.walks = walks
        self.node_attributes = node_attributes
        
        self.hidden_dim = hidden_dim
        self.pos_embed_size = pos_embed_size
        self.feat_size = feat_size
        
        self.pos_embed = pos_embed
        self.hidden_dropout_prob = hidden_dropout_prob
        self.metric = metric
        self.out_activation = out_activation
        
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        
        self._preprocess_data()
        self.reset_model()
        self._embeddings = {}            
    
    def _preprocess_data(self):
        
        if self.pos_embed == 'SVD':
            pos_embed_model = SVD(self.graph, embed_size=self.pos_embed_size)
        elif self.pos_embed == 'LE':
            pos_embed_model = LE(self.graph, self.pos_embed_size)
        
        self.node_pos_embeds = pos_embed_model.get_embeddings()
        self.node_pos_embeds = np.array(list(self.node_pos_embeds.values()))
        if self.node_attributes is not None:
            self.node_attributes = self.node_attributes.to_numpy()

        self.generator = LinkGenerator(self.graph, node_pos_enc=self.node_pos_embeds,
                                       node_attribute=self.node_attributes,
                                       node2idx=self.node2idx)
        
    def reset_model(self):
        
        self.model, self.embed_model = build_model(hidden_dim=self.hidden_dim,
                                                   pos_embed_size=self.pos_embed_size, 
                                                   feat_size=self.feat_size, metric=self.metric, 
                                                   out_activation=self.out_activation)

        def c_loss(margin=1):
            def contrastive_loss(y_true, y_pred):
                y_true = tf.cast(y_true, y_pred.dtype)
                square_pred = tf.math.square(y_pred)
                margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
                return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
            return contrastive_loss
        loss = c_loss(margin=1)
        
        self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=loss,
                metrics=[tf.keras.metrics.AUC(curve="ROC", name="auc_roc")]
            )
    
    def train(self, batch_size=1000, epochs=5, verbose=1):
        unsupervised_sampler = UnsupervisedSampler(self.graph, walks=self.walks)
        batches = unsupervised_sampler.run()
        samples = self.generator.flow([batches[0][:,0], batches[0][:,1]], batched_targets=batches[1])
        
        self.model.fit(
            x=samples[0],
            y=samples[1],
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            )
    
    def get_embeddings(self):
        self._embeddings = {}
        
        node_indices = [self.node2idx[node] for node in self.graph.nodes()]
        if self.node_attributes is not None:
            embeddings = self.embed_model.predict([self.node_pos_embeds[node_indices], self.node_attributes[node_indices]],
                                                  batch_size=self.graph.number_of_nodes())
        else:
            embeddings = self.embed_model.predict(self.node_pos_embeds[node_indices],
                                                  batch_size=self.graph.number_of_nodes())
        
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
        
        return self._embeddings