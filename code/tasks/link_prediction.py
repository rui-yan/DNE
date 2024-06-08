import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.edge_splitter import EdgeSplitter
from ..embedding import NodeEmbedding


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

# ----------------- Link Prediction -----------------
def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

class LinkPredictor(object):
    def __init__(self, args=None, graph=None, results_path=None):
        self.args = args
        self.graph = graph
        self.results_path = results_path
        self.embed_size = getattr(self.args, 'embed_size', 128)
            
    def train_and_evaluate(self, method, node_subjects=None, cv_fold=5, n_trials=5):
        all_score = []
        for trial in range(n_trials):
            seed = trial
            graph_train, examples, labels = split_train_test_edges(self.graph, test_size=0.6, seed=seed)
            X_train, X_test, X_valid = examples
            Y_train, Y_test, Y_valid = labels
            
            embedding_model = NodeEmbedding(self.args, graph_train, "Train Graph")
            
            if node_subjects is not None:
                node_attributes = node_subjects[[col for col in node_subjects.columns if col != 'node_label']].to_numpy()
                embeddings = embedding_model.get_embeddings(method, node_attributes=node_attributes, embed_size=self.embed_size)
            else:
                embeddings = embedding_model.get_embeddings(method, embed_size=self.embed_size)

            self.embeddings = embeddings
            results = []
            for op in binary_operators:
                model = self.train(X_train, Y_train, op, cv_fold)
                valid_score = self.predict(model, X_valid, Y_valid, op)
                results.append(valid_score)
            
            best_result = max(results, key=lambda result: result["score"]["auc_roc"])
            cv_score = self.predict(best_result['classifier'], X_test, Y_test, best_result['binary_operator'])
            score = cv_score['score']
            score['method'] = method
            score['trial'] = trial
            all_score.append(score)
            # print('trial: ', trial, 'score: ', score)
            
            if self.results_path:
                preds_path=f'{self.results_path}/{method}'
                if not os.path.exists(preds_path):
                    os.makedirs(preds_path)
                np.savez(f"{preds_path}/trial_{trial}_results.npz", Y_true=Y_test, Y_pred=cv_score['Y_pred'], Y_prob=cv_score['Y_prob'])
        
        return pd.DataFrame(all_score)
    
    def train(self, X, Y, op, cv_fold):
        lr_clf = LogisticRegressionCV(Cs=10, cv=cv_fold, scoring="roc_auc", max_iter=2000, refit=True)
        clf = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])
        X_embed = op(np.array([self.embeddings[i] for i in np.transpose(X)[0]]),
                     np.array([self.embeddings[j] for j in np.transpose(X)[1]]))
        clf.fit(X_embed, Y)
        return clf

    def predict(self, clf, X, Y, op):
        X_embed = op(np.array([self.embeddings[i] for i in np.transpose(X)[0]]),
                     np.array([self.embeddings[j] for j in np.transpose(X)[1]]))
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
        
        return {"classifier": clf, "binary_operator": op, "score": score, 'Y_pred': Y_pred, 'Y_prob': Y_prob}