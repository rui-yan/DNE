import numpy as np
import networkx as nx
import scipy.sparse as sp
import sys
sys.path.append("..")
from utils.utils_graph import preprocess_nxgraph

class HOPE:
    r"""An implementation of `"HOPE" <https://www.kdd.org/kdd2016/papers/files/rfp0184-ouA.pdf>`
    """
    def __init__(self, graph, embed_size: int = 128, seed: int = 42):

        self.graph = graph
        self.embed_size = embed_size
        self.seed = seed
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
    
    def _create_target(self, graph):
        """
        Creating a target similarity matrix.
        """
        # number_of_nodes = graph.number_of_nodes()
        A = nx.adjacency_matrix(graph)
        S = sp.coo_matrix(A.dot(A), dtype=np.float32)
        return S

    def _do_rescaled_decomposition(self, S):
        """
        Decomposing the similarity matrix.
        """
        U, sigmas, Vt = sp.linalg.svds(S, k=int(self.embed_size / 2))
        sigmas = np.diagflat(np.sqrt(sigmas))
        self._left_embedding = np.dot(U, sigmas)
        self._right_embedding = np.dot(Vt.T, sigmas)
        self._X = np.concatenate([self._left_embedding, self._right_embedding], axis=1)
        
    def fit(self, graph):
        """
        Fitting a HOPE model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        # self._set_seed()
        # graph = self._check_graph(graph)
        S = self._create_target(graph)
        self._do_rescaled_decomposition(S)
    
    def get_embeddings(self):
        r"""Getting the node embedding.

        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
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
        return np.dot(X[i, :self.embed_size // 2], X[j, self.embed_size // 2:])
    
    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "HOPE"