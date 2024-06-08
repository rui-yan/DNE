import numpy as np


class UnsupervisedSampler:
    def __init__(self, G, walks=None):
        
        self.G = G
        self.walks = walks
        self.np_random = np.random.RandomState(42)
    
    def run(self, batch_size=None):
        
        degrees = dict(self.G.degree())
        all_nodes = list(degrees.keys())
        
        sampling_distribution = np.array([degrees[n] ** 0.75 for n in all_nodes])
        sampling_distribution_norm = sampling_distribution / np.linalg.norm(sampling_distribution, ord=1)
        
        # Prepare positive pairs (target, context)
        targets = [walk[0] for walk in self.walks]
        positive_pairs = np.array(
                [
                (target, positive_context)
                for target, walk in zip(targets, self.walks)
                for positive_context in walk[1:]
                ]
            )
        
        # Prepare negative pairs
        negative_samples = self.np_random.choice(all_nodes, size=len(positive_pairs), p=sampling_distribution_norm)
        negative_pairs = np.column_stack((positive_pairs[:, 0], negative_samples))
        
        node_pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
        labels = np.repeat([1, 0], len(positive_pairs))
        
        indices = self.np_random.permutation(len(node_pairs))

        if batch_size is None:
            return (node_pairs[indices], labels[indices])
        else:
            batch_indices = [
                indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
            ]
            return [(node_pairs[i], labels[i]) for i in batch_indices]


class LinkGenerator:
    def __init__(self, graph, node_pos_enc=None, node_attribute=None, node2idx=None):
        self.graph = graph
        self.node_pos_enc = node_pos_enc
        self.node_attribute = node_attribute
        self.node2idx = node2idx if node2idx is not None else {}
    
    def flow(self, batched_node_pairs, batched_targets):
        src_node_indices = [self.node2idx.get(node, -1) for node in batched_node_pairs[0]]
        dst_node_indices = [self.node2idx.get(node, -1) for node in batched_node_pairs[1]]
        
        src_features = self.get_node_features(src_node_indices)
        dst_features = self.get_node_features(dst_node_indices)
        
        batched_node_pairs = [feature for feature in src_features + dst_features if feature is not None]
        
        return batched_node_pairs, batched_targets
    
    def get_node_features(self, node_indices):
        node_pos_embeds = self.node_pos_enc[node_indices] if self.node_pos_enc is not None else None
        node_attributes = self.node_attribute[node_indices] if self.node_attribute is not None else None
        
        return node_pos_embeds, node_attributes