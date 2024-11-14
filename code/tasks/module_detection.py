import os
import random
import numpy as np
import pandas as pd

from multiprocessing import Pool
from collections import defaultdict
from functools import partial

from embedding import NodeEmbedding
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_mutual_info_score

def reduce_module(module, nodes):
    new_module = {}
    for module_name, members in module.items():
        new_members = [node for node in members if node in nodes]
        if len(new_members) > 1:  # don't include empty or single element modules
            new_module[module_name] = new_members
    return new_module

def invert_module(module):
    inverted_module = defaultdict(list)
    for module, node_list in module.items():
        for node in node_list:
            inverted_module[node].append(module)
    return inverted_module

def sample_module(standard, inverted_standard, seed=0):
    """Subsamples the modules in the standard to ensure the resulting module set has
    no overlapping modules (this allows clustering metrics like AMI to be used).
    """
    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    shared_genes = list(inverted_standard.keys())
    shuffled_genes = np.random.choice(shared_genes, size=len(shared_genes), replace=False)

    # track newly sampled standard and sampled genes
    sampled_standard = {}
    sampled_genes = set()

    for label, gene in enumerate(shuffled_genes):

        # if `gene` has already been sampled, skip it
        if gene in sampled_genes:
            continue
        
        sampled_module = random.sample(inverted_standard[gene], 1)[0]
        sampled_module_genes = standard[sampled_module]

        # check for overlaps
        in_sampled_genes = [
            True if gene_ in sampled_genes else False for gene_ in sampled_module_genes
        ]

        # if any overlaps exist (i.e. `gene` exists in another module), ignore current
        # `sampled_module`
        if any(in_sampled_genes):
            continue

        # record genes in sampled module and assign these genes to a module label
        for gene_ in sampled_module_genes:
            sampled_genes.add(gene_)
            sampled_standard[gene_] = label

    sampled_genes = list(sampled_genes)
    standard_labels = [sampled_standard[gene_] for gene_ in sampled_genes]
    return sampled_genes, standard_labels

def predict(features, labels):
    best_cluster_labels = None
    best_score = {'ami': 0, "linkage": None, "metric": None, "threshold": None}
    
    # Constants
    methods = ["average", "single", "complete"]
    metrics = ["cosine", "euclidean"]
    num_thresholds = 20
    
    # Iterate over parameter combinations and identify the best scoring module set
    for method in methods:
        for metric in metrics:
            print(f'\n {method}, {metric}')
            
            features_ = squareform(pdist(features, metric=metric))
            link = linkage(features_, method=method)
            
            thresholds = np.linspace(0, np.max(link[:, 2]), num=num_thresholds)
            
            for t in thresholds:
                cluster_labels = fcluster(link, t, criterion='distance')
                
                score = {
                    'ami': adjusted_mutual_info_score(labels, cluster_labels),
                    'linkage': method,
                    'metric': metric,
                    'threshold': t
                }
                
                # Update best score and corresponding cluster labels
                if score['ami'] > best_score['ami']:
                    best_score = score
                    best_cluster_labels = cluster_labels
    
    return best_score, best_cluster_labels

# ----------------- module detection -----------------
class ModuleDetector(object):
    def __init__(self, args, graph, results_path=None):
        self.args = args
        self.graph = graph
        self.results_path = results_path
    
    def save_embeddings(self, method, embeddings):
        df = pd.DataFrame({key: value.tolist() for key, value in embeddings.items()})
        csv_filename = f'{self.results_path}/{method}_features.csv'
        df.T.to_csv(csv_filename, index=True)
    
    def train_and_evaluate(self, modules, method, node_subjects=None,  n_trials=10):
        module = reduce_module(modules, nodes=set(self.graph.nodes()))
        inverted_module = invert_module(module)
        
        embedding_model = NodeEmbedding(self.args, self.graph, "Whole Graph")
        if node_subjects is not None:
            node_attributes = node_subjects[[col for col in node_subjects.columns if col != 'node_label']]
            embeddings = embedding_model.get_embeddings(method, node_attributes=node_attributes, 
                                                    embed_size=self.args.embed_size)
        else:
            embeddings = embedding_model.get_embeddings(method, embed_size=self.args.embed_size)
        self.save_embeddings(method, embeddings)
        
        # evaluate features and networks using multiprocessing
        with Pool() as p:
            _evaluate_partial = partial(self._evaluate, [module, inverted_module, embeddings, method])
            results = p.map(_evaluate_partial, range(n_trials))
        
        return pd.DataFrame(results)
    
    def _evaluate(self, args, trial):
        """Wrapper function for running evaluations in in a single process."""
        module, inverted_module, embeddings, method = args
        sampled_genes, labels = sample_module(module, inverted_module, seed=trial) #4242 * trial
        features = np.asarray([embeddings[x] for x in sampled_genes])
        
        score, cluster_labels = predict(features, labels)
        score['trial'] = trial
        score['method'] = method
        
        if self.results_path:
            preds_path=f'{self.results_path}/{method}'
            if not os.path.exists(preds_path):
                os.makedirs(preds_path)
            
            np.savez(f"{preds_path}/trial_{trial}_results.npz", index=np.array(sampled_genes), Y_true=np.array(labels), Y_pred=cluster_labels)
        
        return score