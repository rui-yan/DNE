import os
import random
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tasks.link_prediction import LinkPredictor
from tasks.link_prediction_heuristic import LinkPredictorHeuristic
from tasks.module_detection import ModuleDetector
from dataset import GraphDataset

# Define metrics
LINK_PREDICTION_METRICS = ["auc_roc", "auc_pr", "f1", "acc", "bcc"]
MODULE_DETECTION_METRICS = ["ami"] 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='a_thaliana', type=str, choices=[
                        'a_thaliana', 'c_elegans', 'HuRI', 's_cerevisiae', 'cora', 'Power', 'Router'
                        ], help='dataset name')
    parser.add_argument('--task', default='link_prediction', type=str, choices=[
                        'link_prediction', 'module_detection',
                        ], help='task to perform')
    parser.add_argument('--task_label', default=None, type=str, choices=[
                        'GOBP', 'IntAct', 'KEGG'
                        ], help='labels for module identification')
    parser.add_argument('--add_feats', action='store_true', default=False, help='use node features')
    parser.add_argument('--n_trials', default=1, type=int, help='number of trials')
    
    # model related
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=5120, type=int, help='batch size')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    parser.add_argument('--embed_size', default=128, type=int, help='embedding size')
    
    # random walk related
    parser.add_argument('--walk_number', default=100, type=int, 
                        help='number of random walks per node')
    parser.add_argument('--walk_length', default=10, type=int, 
                        help='length of each random walk')
    parser.add_argument('--p', default=1.0, type=float,
                        help='p controls how fast the walk explores')
    parser.add_argument('--q', default=1.0, type=float,
                        help='q controls how fast the walk leaves the neighborhood of starting node')
    
    # others
    parser.add_argument('--data_path', default='../data', help='path to data')
    parser.add_argument('--save_path', default='../result', help='path to save results')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    
    return args


def main(args):
    # Load dataset
    graph_data = GraphDataset(args.data_path)
    graph_data.load_graph(args.dataset, add_feats=args.add_feats)
    graph, node_subjects = graph_data.graph, graph_data.node_subjects
    
    if node_subjects.empty:
        node_subjects = None
    
    # print(f'Sample of node IDs: {list(graph.nodes())[:5]}')
    # print('node_subjects: ', node_subjects)
    
    # Calculate graph statistics
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()
    edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2)
    
    # Calculate average degree
    degree_sequence = list(dict(graph.degree()).values())
    average_degree = sum(degree_sequence) / num_nodes
    
    # Print graph statistics
    print("\nGraph Loaded:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Edge density: {edge_density}")
    print(f"Average degree: {average_degree}")
    
    if args.task == 'link_prediction_heuristic':
        methods = ['JC', 'CN', 'PA', 'RA', 'RP', 'Katz']
    else:
        # methods = ['DNE'] 
        methods = ['DNE', 'GraRep', 'HOPE', 'NetMF', 'LLE', 'N2V', 'SVD']
    
    results_path = None
    if args.task == 'link_prediction':
        results_path=f'{args.save_path}/{args.task}/{args.dataset}/'
    elif args.task == 'link_prediction_heuristic':
        results_path=f'{args.save_path}/{args.task}/{args.dataset}/'
    elif args.task == 'module_detection':
        results_path=f'{args.save_path}/{args.task}/{args.dataset}/{args.task_label}'
    
    if results_path:
        eval_result_file=f'{results_path}/result.txt'
        eval_avg_result_file=f'{results_path}/avg_result.txt'
        
        if os.path.exists(eval_result_file):
            os.remove(eval_result_file)
    
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    
    df_result = pd.DataFrame()
    if args.task == 'link_prediction':
        metrics = LINK_PREDICTION_METRICS
        clf = LinkPredictor(args=args, graph=graph, results_path=results_path)
        for method in methods:
            print(f"\n---- {method} link prediction ----")
            result = clf.train_and_evaluate(method, node_subjects, cv_fold=5, n_trials=args.n_trials)
            df_result = pd.concat([df_result, result])
    
    elif args.task == 'link_prediction_heuristic':
        metrics = LINK_PREDICTION_METRICS
        clf = LinkPredictorHeuristic(args=args, graph=graph, results_path=results_path)
        for method in methods:
            print(f"\n---- {method} link prediction using heuristic methods ----")
            result = clf.train_and_evaluate(method, cv_fold=5, n_trials=args.n_trials)
            df_result = pd.concat([df_result, result])
    
    elif args.task == 'module_detection':
        import json
        metrics = MODULE_DETECTION_METRICS
        clf = ModuleDetector(args=args, graph=graph, results_path=results_path)
        module_base_path = '/home/yan/DNE/data/s_cerevisiae/standards/module-detection/'
        if args.task_label == 'GOBP':
            module_fname = os.path.join(module_base_path, "yeast-GO-bioprocess-modules.json")
        elif args.task_label == 'IntAct':
            module_fname = os.path.join(module_base_path, "yeast-IntAct-complex-modules.json")
        elif args.task_label == 'KEGG':
            module_fname = os.path.join(module_base_path, "yeast-KEGG-pathway-modules.json")
        with open(module_fname, "r") as f:
            modules = json.load(f)
        for method in methods:
            print(f"\n---- {method} module detection ----")
            result = clf.train_and_evaluate(modules, method, node_subjects, n_trials=args.n_trials)
            df_result = pd.concat([df_result, result])

    parent_directory = os.path.dirname(eval_result_file)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    
    df_result.to_csv(eval_result_file, sep='\t', index=False)
    avg_result = df_result.groupby(["method"])[metrics].agg(["mean", "std"]).reset_index()
    for metric in metrics:
        avg_result[(metric, "mean")] = avg_result[(metric, "mean")].round(4)
        avg_result[(metric, "std")] = avg_result[(metric, "std")].round(4)
    avg_result.to_csv(eval_avg_result_file, sep='\t')
    # print("\nMean Performance Metrics:")
    # print(avg_result.to_string(index=False), '\n')  


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    main(args)