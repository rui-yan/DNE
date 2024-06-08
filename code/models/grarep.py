import math
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from ..utils.utils_graph import preprocess_nxgraph

class GraRep:
    """
    GraRep Model Object.
    A sparsity aware implementation of GraRep.
    For details see the paper: https://dl.acm.org/citation.cfm?id=2806512
    """
    def __init__(self, graph, embed_size=128, seed=42):
        """
        :param A: Adjacency matrix.
        :param args: Arguments object.
        """
        self.A = nx.to_scipy_sparse_array(graph, format='coo', dtype=float)
        self.embed_size = embed_size
        self.seed = seed

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self._setup_base_target_matrix()

    def _setup_base_target_matrix(self):
        """
        Creating a base matrix to multiply.
        """
        values = [1.0 for i in range(self.A.shape[0])]
        indices = [i for i in range(self.A.shape[0])]
        self.A_hat = sparse.coo_matrix((values, (indices, indices)),
                                        shape=self.A.shape,
                                        dtype=np.float32)

    def _create_target_matrix(self):
        """
        Creating a log transformed target matrix.
        :return target_matrix: Matrix to decompose with SVD.
        """
        self.A_hat = sparse.coo_matrix(self.A_hat.dot(self.A))
        scores = np.log(self.A_hat.data)-math.log(self.A.shape[0])
        rows = self.A_hat.row[scores < 0]
        cols = self.A_hat.col[scores < 0]
        scores = scores[scores < 0]
        target_matrix = sparse.coo_matrix((scores, (rows, cols)),
                                           shape=self.A.shape,
                                           dtype=np.float32)
        return target_matrix

    def train(self, order):
        """
        Learning an embedding.
        """
        embeddings = []
        for _ in range(order):
            target_matrix = self._create_target_matrix()

            svd = TruncatedSVD(n_components=int(self.embed_size / order),
                               n_iter=20,
                               random_state=self.seed)

            svd.fit(target_matrix)
            embedding = svd.transform(target_matrix)
            embeddings.append(embedding)
        self._X = np.concatenate(embeddings, axis=1)

    def get_embeddings(self):
        self._embeddings = {}

        idx2node = self.idx2node
        for i, embedding in enumerate(self._X):
            self._embeddings[idx2node[i]] = embedding

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
        return "GraRep"