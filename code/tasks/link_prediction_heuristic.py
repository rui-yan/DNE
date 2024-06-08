import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
import functools

from utils.edge_splitter import EdgeSplitter
from .similarities import *


def split_train_test_edges(graph, test_size, seed):

    # Define an edge splitter on the original graph:
    edge_splitter_test = EdgeSplitter(graph)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, X_test, Y_test = edge_splitter_test.train_test_split(
        p=0.1, 
        keep_connected=False, 
        seed=seed
    )
    
    # Do the same process to compute a training subset from within the test graph
    edge_splitter_train = EdgeSplitter(graph_test)
    graph_train, X, Y = edge_splitter_train.train_test_split(
        p=0.1, 
        keep_connected=False, 
        seed=seed
    )
    (
        X_train,
        X_valid,
        Y_train,
        Y_valid,
    ) = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # print('len(labels_train): ', len(Y_train))
    # print('len(labels_test): ', len(Y_test))
    # print('len(labels_valid):', len(Y_valid))
    
    examples = (X_train, X_test, X_valid)
    labels = (Y_train, Y_test, Y_valid)

    return graph_train, examples, labels

def binary_to_neighborhood(br):
    relation = defaultdict(set)
    for nodeArr in br:
        relation[nodeArr[0]].add(nodeArr[1])
        relation[nodeArr[1]].add(nodeArr[0])
    return relation


def compute_similarity_matrix(graph, edges, method='AA'):
    @functools.lru_cache(maxsize=None)
    def RP_calc(u):
        return rooted_pagerank_score(graph,u)
    
    def RP_compute(u,v):
        return RP_calc(u)[v] + RP_calc(v)[u]
    
    if method == 'AA':
        scores = [AA_score(graph,u,v) for (u,v) in edges]
    elif method == 'JC':
        scores = [JC_score(graph,u,v) for (u,v) in edges]
    elif method == 'CN':
        scores = [CN_score(graph,u,v) for (u,v) in edges]
    elif method == 'PA':
        scores = [PA_score(graph,u,v) for (u,v) in edges]
    elif method == 'RA':
        scores = [RA_score(graph,u,v) for (u,v) in edges]
    elif method == 'RP':
        scores = [RP_compute(u,v) for (u,v) in edges]
    elif method == 'Katz':
        nodes = list(graph.nodes)
        katz_matrix = katz_score(graph)
        scores = [katz_matrix[nodes.index(u), nodes.index(v)] for (u,v) in edges]
    
    scores = np.array(scores).reshape(len(scores), -1)

    return scores

# ----------------- Link Prediction -----------------
class LinkPredictorHeuristic(object):
    def __init__(self, args, graph, results_path=None):
        self.args = args
        self.graph = graph
        self.results_path = results_path
    
    def train_and_evaluate(self, method, cv_fold=5, n_trials=5):
        all_score = []
        
        for trial in range(n_trials):
            seed = trial
            graph_train, examples, labels = split_train_test_edges(self.graph, test_size=0.6, seed=seed)
            X_train, X_test, X_valid = examples
            Y_train, Y_test, Y_valid = labels
            
            results = []
            model = self.train(X_train, Y_train, graph_train, cv_fold, method)
            valid_score = self.predict(model, X_valid, Y_valid, graph_train, method)
            results.append(valid_score)
            
            best_result = max(results, key=lambda result: result["score"]["auc_roc"])
            cv_score = self.predict(best_result['classifier'], X_test, Y_test, graph_train, method)
            score = cv_score['score']

            score['method'] = method
            score['trial'] = trial
            all_score.append(score)
            
            if self.results_path:
                # print(f'{self.results_path}/trial_{trial}_results.npz')
                preds_path=f'{self.results_path}/{method}'
                # Check if the parent directory exists, and create it if not
                if not os.path.exists(preds_path):
                    os.makedirs(preds_path)
                np.savez(f"{preds_path}/trial_{trial}_results.npz", Y_true=Y_test, Y_pred=cv_score['Y_pred'], Y_prob=cv_score['Y_prob'])
        
        return pd.DataFrame(all_score)
    
    def train(self, X, Y, graph, cv_fold, method):
        clf = LogisticRegressionCV(Cs=10, cv=cv_fold, scoring="roc_auc", max_iter=2000, refit=True)
        edges_train = X
        X_embed = compute_similarity_matrix(graph, edges_train, method)
        clf.fit(X_embed, Y)
        return clf

    def predict(self, clf, X, Y, graph, method):
        edges = X
        X_embed = compute_similarity_matrix(graph, edges, method)
        Y_prob = clf.predict_proba(X_embed)[:, 1]
        Y_pred = clf.predict(X_embed)

        # Calculate various metrics
        score = {
            'auc_roc': round(roc_auc_score(Y, Y_prob), 4),
            'auc_pr': round(average_precision_score(Y, Y_prob), 4),
            'acc': round(accuracy_score(Y, Y_pred), 4),
            'f1': round(f1_score(Y, Y_pred), 4),
            'bcc': round(balanced_accuracy_score(Y, Y_pred), 4)
        }
        
        return {"classifier": clf, "score": score, 'Y_pred': Y_pred, 'Y_prob': Y_prob}