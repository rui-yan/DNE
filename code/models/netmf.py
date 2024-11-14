import numpy as np
import networkx as nx
from scipy import sparse
import sys
sys.path.append("..")
from utils.utils_graph import preprocess_nxgraph

class NetMF:
    r"""An implementation of `"NetMF" <https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf>`
    """
    def __init__(
        self,
        graph,
        embed_size: int = 32,
        window_size: int = 5,
        rank: int = 1,
        negative_samples: int = 1,
        is_large: bool = False,
        seed: int = 0,
    ):
        self.graph = graph
        self.embed_size = embed_size
        self.window_size = window_size
        self.rank = rank
        self.negative_samples = negative_samples
        self.is_large = is_large
        self.seed = seed
        
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.learn_embedding()
    
    def learn_embedding(self, return_dict=False):
        A = sparse.csr_matrix(nx.adjacency_matrix(self.graph))
        if not self.is_large:
            print("Running NetMF for a small window size...")
            deepwalk_matrix = self._compute_deepwalk_matrix(A, window=self.window_size, b=self.negative_samples)
        else:
            print("Running NetMF for a large window size...")
            vol = float(A.sum())
            evals, D_rt_invU = self._approximate_normalized_laplacian(A, rank=self.rank, which="LA")
            deepwalk_matrix = self._approximate_deepwalk_matrix(
                evals, D_rt_invU, window=self.window_size, vol=vol, b=self.negative_samples
            )
        # factorize deepwalk matrix with SVD
        u, s, _ = sparse.linalg.svds(deepwalk_matrix, self.embed_size)
        self._X = sparse.diags(np.sqrt(s)).dot(u.T).T
    
    def _compute_deepwalk_matrix(self, A, window, b):
        # directly compute deepwalk matrix
        n = A.shape[0]
        vol = float(A.sum())
        L, d_rt = sparse.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} A D^{-1/2}
        X = sparse.identity(n) - L
        S = np.zeros_like(X)
        X_power = sparse.identity(n)
        for i in range(window):
            print("Compute matrix %d-th power", i + 1)
            X_power = X_power.dot(X)
            S += X_power
        S *= vol / window / b
        D_rt_inv = sparse.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T).todense()
        M[M <= 1] = 1
        Y = np.log(M)
        return sparse.csr_matrix(Y)

    def _approximate_normalized_laplacian(self, A, rank, which="LA"):
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2} and keep top rank eigenpairs
        n = A.shape[0]
        L, d_rt = sparse.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} W D^{-1/2}
        X = sparse.identity(n) - L
        print("Eigen decomposition...")
        evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
        print("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
        print("Computing D^{-1/2}U..")
        D_rt_inv = sparse.diags(d_rt ** -1)
        D_rt_invU = D_rt_inv.dot(evecs)
        return evals, D_rt_invU

    def _deepwalk_filter(self, evals, window):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1.0 if x >= 1 else x * (1 - x ** window) / (1 - x) / window
        evals = np.maximum(evals, 0)
        print(
            "After filtering, max eigenvalue=%f, min eigenvalue=%f", np.max(evals), np.min(evals),
        )
        return evals

    def _approximate_deepwalk_matrix(self, evals, D_rt_invU, window, vol, b):
        # approximate deepwalk matrix
        evals = self._deepwalk_filter(evals, window=window)
        X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
        M = X.dot(X.T) * vol / b
        M[M <= 1] = 1
        Y = np.log(M)
        print("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
        return sparse.csr_matrix(Y)
    
    def get_embeddings(self):
        self._embeddings = {}

        idx2node = self.idx2node
        for i, embedding in enumerate(self._X):
            self._embeddings[idx2node[i]] = embedding

        return self._embeddings