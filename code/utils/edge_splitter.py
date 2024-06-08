import numpy as np
import networkx as nx


class EdgeSplitter(object):
    def __init__(self, g):
        """
        Initialize the EdgeSplitter.

        Args:
            g: The original graph to be copied.
        """
        self.g = g.copy()
        self.g_train = None
        self._random = None

    def _train_test_split_homogeneous(self, p, keep_connected=False):
        """
        Method for edge splitting applied to homogeneous graphs.

        Args:
            p (float): Percent of edges to be returned.
            keep_connected (bool): If True, ensure the reduced graph remains connected.

        Returns:
            Tuple of two numpy arrays: the first Nx2 holding the node ids for the edges
            and the second Nx1 holding the edge labels, 0 for negative and 1 for positive example.
        """
        if keep_connected:
            min_edges = self._get_minimum_spanning_edges()
        else:
            min_edges = set()
        
        # Sample the positive examples
        positive_edges = self._reduce_graph(min_edges, p)
        positive_edges = np.array(positive_edges)

        # Sample the negative examples
        negative_edges = self._sample_negative_examples_global(p, limit_samples=len(positive_edges))
        negative_edges = np.array(negative_edges)

        edge_data_ids = np.vstack((positive_edges[:, :2], negative_edges[:, :2]))
        edge_data_labels = np.hstack((np.ones(len(positive_edges)), np.zeros(len(negative_edges))))

        # print("** Sampled {} positive and {} negative edges. **".format(len(positive_edges), len(negative_edges)))
        return edge_data_ids, edge_data_labels
    
    def _get_minimum_spanning_edges(self):
        """
        Calculate the minimum set of edges such that graph connectivity is preserved.

        Returns:
            List of the minimum spanning edges of the undirected graph self.g.
        """
        if self.g.is_directed():
            mst = nx.minimum_branching(self.g).edges
        else:
            mst = nx.minimum_spanning_edges(self.g, data=False)        
        
        edges = list(mst)
        min_edges_set = {(u[0], u[1]) for u in edges}
        min_edges_set.update({(u[1], u[0]) for u in edges}) 

        return min_edges_set
    
    def train_test_split(
        self,
        p: float = 0.5,
        probs=None,
        keep_connected: bool = False,
        seed: int = None,
    ):
        """
        Generates positive and negative edges and a reduced graph with positive edges removed.

        Args:
            p (float): Percent of edges to be returned.
            probs (list): Probabilities for sampling a node that is k-hops from the source node.
            keep_connected (bool): If True, ensure the reduced graph remains connected.
            seed (int): Seed for random number generator.

        Returns:
            Tuple containing the reduced graph (positive edges removed), edge data as two numpy arrays.
        """
        if p <= 0 or p >= 1:
            raise ValueError("The value of p must be in the interval (0, 1)")

        if self._random is None:
            self._random = np.random.RandomState(seed=seed)

        edge_data_ids, edge_data_labels = self._train_test_split_homogeneous(
            p=p, keep_connected=keep_connected
        )

        result_graph = self.g_train
        return result_graph, edge_data_ids, edge_data_labels
    
    def _reduce_graph(self, min_edges, p: float):
        """
        Reduces the graph by removing existing edges not in min_edges.

        Args:
            min_edges (set): Minimum spanning tree edges that cannot be removed.
            p (float): Factor by which to reduce the size of the graph.

        Returns:
            List of edges removed from self.g_train.
        """
        self.g_train = self.g.copy()

        all_edges = list(self.g_train.edges())
        num_edges_to_remove = int(self.g_train.number_of_edges() * p)

        if num_edges_to_remove > (self.g_train.number_of_edges() - len(min_edges)):
            raise ValueError(
                "Not enough positive edges to sample after reserving {} number of edges for maintaining graph connectivity. Consider setting keep_connected=False.".format(
                    len(min_edges)
                )
            )

        self._random.shuffle(all_edges)
        removed_edges = []
        for edge in all_edges:
            if edge not in min_edges:
                removed_edges.append((edge[0], edge[1], 1))
                self.g_train.remove_edge(*edge)

                if len(removed_edges) == num_edges_to_remove:
                    break

        return removed_edges
    
    def _sample_negative_examples_global(self, p: float, limit_samples=None):
        num_edges_to_sample = int(self.g.number_of_edges() * p)

        if limit_samples is not None:
            num_edges_to_sample = min(num_edges_to_sample, limit_samples)

        nodes = list(self.g.nodes(data=False))
        num_nodes = len(nodes)
        num_neg_samples = min(num_edges_to_sample, num_nodes * (num_nodes - 1) // 2)

        if self.g.is_directed():
            all_edges = set(self.g.edges())
        else:
            all_edges = set(self.g.edges())
            all_edges.update({(u[1], u[0]) for u in all_edges})

        sampled_edges = []
        while len(sampled_edges) < num_neg_samples:
            num_samples_needed = num_neg_samples - len(sampled_edges)

            if num_samples_needed > num_nodes * (num_nodes - 1) // 2:
                # Reduce the number of samples to the maximum possible
                num_samples_needed = num_nodes * (num_nodes - 1) // 2

            sampled_pairs = self._random.choice(nodes, size=(num_samples_needed, 2), replace=True)

            for u, v in sampled_pairs:
                if (u, v) not in all_edges and (v, u) not in all_edges:
                    sampled_edges.append((u, v, 0))

        return sampled_edges