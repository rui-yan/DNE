import numpy as np
from gensim.models import Word2Vec
import sys
sys.path.append("..")
from utils.walker import RandomWalker

class Node2Vec:
    r"""An implementation of `"Node2Vec" <https://arxiv.org/abs/1607.00653>`
    """
    def __init__(self, graph, embed_size=128, seed=42):
        self.graph = graph
        self._embeddings = {}
        self.embed_size = embed_size

    def _simulate_walks(self, n, l, p, q, use_rejection_sampling=True):
        walker = RandomWalker(self.graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)
        walker.preprocess_transition_probs()
        walks = walker.simulate_walks(num_walks=n, walk_length=l, workers=1, verbose=1)
        return walks
    
    def train(self, window_size=5, workers=3, iter=5, walk_number=100, walk_length=10, p=1.0, q=1.0, **kwargs):
        # Simulate random walks
        walks = self._simulate_walks(n=walk_number, l=walk_length, p=p, q=q)

        kwargs["sentences"] = walks
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = self.embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        model = Word2Vec(**kwargs)
        self.w2v_model = model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def get_reconstructed_adj(self, X=None, node_l=None):
        """Compute the adjacency matrix from the learned embedding

        Returns:
            A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
        """
        node_num = self._X.shape[0]
        
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        
        return adj_mtx_r
    
    def get_edge_weight(self, i, j):
        """Compute the weight for edge between node i and node j

        Args:
            i, j: two node id in the graph for embedding
        Returns:
            A single number represent the weight of edge between node i and node j

        """
        X = self._embeddings.values()
        return np.dot(X[i, :], X[j, :])
        
    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Node2Vec"