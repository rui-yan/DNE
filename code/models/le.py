import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from ..utils.utils_graph import preprocess_nxgraph

class LE:
    def __init__(self, graph, embed_size=128, seed=42):
        self.graph = graph
        self._embeddings = {}

        self.embed_size = embed_size
        self.seed = seed

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.learn_embedding()
    
    def compute_normalized_laplacian(self, graph):
        # L = sp.coo_matrix(nx.normalized_laplacian_matrix(graph))
        A = nx.adjacency_matrix(graph).astype(float)
        degrees = dict(graph.degree())
        D_inv_sqrt = sp.diags(1 / np.sqrt(list(degrees.values())).clip(1), dtype=float)
        L = sp.eye(graph.number_of_nodes()) - D_inv_sqrt @ A @ D_inv_sqrt
        return L
    
    def learn_embedding(self):
        # graph = self.graph.to_undirected()
        # L = nx.normalized_laplacian_matrix(graph)
        # w, v = lg.eigs(L, k=self.embed_size + 1, which='SM')
        # idx = np.argsort(w)  # sort eigenvalues
        # w = w[idx]
        # v = v[:, idx]
        # self._X = v[:, 1:].real
        
        # p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        # eig_err = np.linalg.norm(p_d_p_t - l_sym)
        # print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        
        L = self.compute_normalized_laplacian(self.graph)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        # EigVal, EigVec = sp.linalg.eigs(L, k=self.embed_size+1, which='SR', tol=1e-2)
        # EigVec = EigVec[:, EigVal.argsort()]
        self._X = EigVec[:,1:self.embed_size+1]
    
    def get_embeddings(self):
        self._embeddings = {}

        idx2node = self.idx2node
        for i, embedding in enumerate(self._X):
            self._embeddings[idx2node[i]] = embedding
        
        return self._embeddings
    
    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Laplacian Eigenmaps"
