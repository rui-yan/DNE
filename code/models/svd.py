import networkx as nx
import scipy.sparse as sp
import numpy as np

import sys
sys.path.append('..')
from utils.utils_graph import preprocess_nxgraph

def compute_normalized_laplacian(graph):
    # L = sp.coo_matrix(nx.normalized_laplacian_matrix(graph))
    A = nx.adjacency_matrix(graph).astype(float)
    degrees = dict(graph.degree())
    D_inv_sqrt = sp.diags(1 / np.sqrt(list(degrees.values())).clip(1), dtype=float)
    L = sp.eye(graph.number_of_nodes()) - D_inv_sqrt @ A @ D_inv_sqrt
    return L
        
class SVD:
    def __init__(self, graph, input='adjacency', embed_size=128, seed=42):
        self.graph = graph
        self._embeddings = {}
        self.input = input
        self.embed_size = embed_size
        self.seed = seed

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.learn_embedding()
    
    def learn_embedding(self):
        if self.input == 'adjacency':
            M = nx.to_scipy_sparse_array(self.graph, format='csr', dtype=float)
        elif self.input == 'laplacian':
            M = compute_normalized_laplacian(self.graph)
        u, s, vt = sp.linalg.svds(M, k=self.embed_size)
        s = np.diag(s)
        X1 = np.dot(u, np.sqrt(s))
        X2 = np.dot(vt.T, np.sqrt(s))
        self._X = X1 + X2
    
    def get_embeddings(self):
        self._embeddings = {}

        idx2node = self.idx2node
        for i, embedding in enumerate(self._X):
            self._embeddings[idx2node[i]] = embedding
        
        return self._embeddings

# A = nx.to_scipy_sparse_array(self.graph, format='csr', dtype=float)
# u, s, _ = sp.linalg.svds(A, k=self.embed_size)
# self._X = sp.diags(np.sqrt(s)).dot(u.T).T

# svd = TruncatedSVD(n_components=self.embed_size,
#                     n_iter=20,
#                     random_state=self.seed)
# svd.fit(A)
# self._X = svd.transform(A)