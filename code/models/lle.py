import networkx as nx
import scipy.sparse as sp
import numpy as np
from ..utils.utils_graph import preprocess_nxgraph
from sklearn.preprocessing import normalize


class LLE:
    def __init__(self, graph, embed_size=128, seed=42):
        self.graph = graph
        self._embeddings = {}
        self.input = input
        self.embed_size = embed_size
        self.seed = seed

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.learn_embedding()
    
    def learn_embedding(self, is_weighted=False):
        graph = self.graph.to_undirected()
        A = nx.to_scipy_sparse_array(graph, format='csr', dtype=float)
        normalize(A, norm='l1', axis=1, copy=False)
        i_n = sp.eye(len(graph.nodes))
        i_min_A = (i_n - A).T.dot(i_n - A)
        w, v = np.linalg.eig(i_min_A.toarray())
        
        # 'SM' finds the smallest magnitude eigenvalues which corresponds to the smallest eigenvalues
        # Since eigsh doesn't guarantee sorted order, we sort the eigenvalues and eigenvectors
        idx = w.real.argsort() # increasing order
        v = v[:, idx]
        # v = np.real(v[:, idx])
        
        # Skip the first eigenvector and take the next 'dim' eigenvectors
        self._X = v.real[:,1:self.embed_size+1]
        
        # A = cp.asarray(nx.to_numpy_array(self.graph, nodelist=self.graph.nodes(), weight='weight'))
        # # Manually compute L1 normalization along axis 1 (rows)
        # row_sums = cp.sum(A, axis=1)
        # A /= row_sums.reshape(-1, 1)
        # I_n = cp.eye(self.graph.number_of_nodes())
        # I_min_A = cp.dot((I_n - A).T, (I_n - A))
        # w, v = cp.linalg.eigh(I_min_A)
        # idx = cp.argsort(w.real)
        # v = v[:, idx]
        # embedding = v[:, 1:(self.embed_size+1)]
        # self._X = embedding.get().real 
    
    def get_embeddings(self):
        self._embeddings = {}

        idx2node = self.idx2node
        for i, embedding in enumerate(self._X):
            self._embeddings[idx2node[i]] = embedding
        
        return self._embeddings
