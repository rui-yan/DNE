import os 
import pandas as pd
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx


class GraphDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.graph = None
        self.node_subjects = None
    
    def load_graph(self, dataset, add_feats=False):
        if dataset in ['a_thaliana', 'c_elegans', 'HuRI']:
            self.edge_fname = os.path.join(self.base_path, f"{dataset}/edge_list.csv")
            graph = read_graph(
                edge_fname=self.edge_fname, 
                edge_list_separator=",",
                edge_list_header=True, 
                edge_src_col=0,
                edge_dst_col=1,
                edge_list_numeric_node_ids=False,
                edge_weight_col=None,
                directed=False,
                name=dataset,
                )
            node_subjects = pd.DataFrame()
        
        elif dataset == 's_cerevisiae':
            self.edge_fname = os.path.join(self.base_path, f"{dataset}/edge_list.txt")
            graph = nx.read_weighted_edgelist(self.edge_fname, delimiter=' ')
            if add_feats:
                node_subjects = pd.read_csv(f'{self.base_path}/{dataset}/Krogan-2006_esm_emb_t36.csv', index_col=0)
                node_subjects.set_index("node_id", inplace=True)
                node_subjects = node_subjects.loc[list(graph.nodes()),:]
            else:
                node_subjects = pd.DataFrame()
                
        elif dataset == 'cora':
            dataset = Planetoid(root=self.base_path, name=dataset)
            data = dataset[0]
            graph = to_networkx(data, to_undirected=True)
            if add_feats:
                node_subjects = pd.DataFrame(data.x, index=list(graph.nodes()))
            else:
                node_subjects = pd.DataFrame()
        
        elif dataset in ['USAir', 'NS', 'PB', 'Power', 'Router']:
            data_dir = os.path.join(self.base_path, 'others', f'{dataset}.mat')
            import scipy.io as sio
            net = sio.loadmat(data_dir)
            graph = nx.from_scipy_sparse_array(net['net'])
            node_subjects = pd.DataFrame()
        
        self.graph = graph
        self.node_subjects = node_subjects
    
    def normalize(self):
        weights = nx.get_edge_attributes(self.graph, 'weight')
        min_weight = min(weights.values())
        max_weight = max(weights.values())

        for edge, weight in weights.items():
            normalized_weight = (weight - min_weight) / (max_weight - min_weight)
            self.graph[edge[0]][edge[1]]['weight'] = normalized_weight        
    
    def filter_edges_by_weight(self, min_edge_weight):
        edges_to_remove = [(u, v) for u, v, weight in self.graph.edges(data='weight') if weight < min_edge_weight]
        self.graph.remove_edges_from(edges_to_remove)
        self.update_node_subjects()
    
    def remove_self_loops(self):
        self_loop_nodes = [node for node in self.graph.nodes() if self.graph.has_edge(node, node)]
        self.graph.remove_nodes_from(self_loop_nodes)
        self.update_node_subjects()
    
    def remove_disconnected_nodes(self):
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        self.update_node_subjects()
    
    def keep_only_largest_connected_components(self):
        largest_component = max(nx.connected_components(self.graph), key=len)
        nodes_to_remove = [node for node in self.graph.nodes() if node not in largest_component]
        self.graph.remove_nodes_from(nodes_to_remove)
        self.update_node_subjects()
    
    def update_node_subjects(self):
        graph_nodes = set(self.graph.nodes())
        self.node_subjects = self.node_subjects[self.node_subjects.node_name.isin(graph_nodes)]


def read_graph(edge_fname, 
               edge_list_separator='\t',
               edge_list_header=False,
               edge_src_col=0,
               edge_dst_col=1,
               edge_list_numeric_node_ids=True,
               edge_weight_col=None,
               edge_type_col=None,
               edge_type_name=None,
               directed=False,
               name=None):
    '''
    Reads the input network in networkx.

    Parameters:
    - edge_fname: Path to the file containing the edge list.
    - edge_list_separator: Separator used in the edge file.
    - edge_list_header: Whether the edge file has a header.
    - edge_src_col: Index of the source column in the edge file.
    - edge_dst_col: Index of the destination column in the edge file.
    - edge_list_numeric_node_ids: Both src and dst columns use numeric node_ids instead of node names
    - edge_weight_col: Index of the weight column in the edge file.
    - directed: Whether the graph is directed.
    - name: Name to assign to the graph.

    Returns:
    - G: NetworkX graph.
    - node_df: Pandas DataFrame containing node information.
    '''
    G = nx.DiGraph() if directed else nx.Graph()
    with open(edge_fname, 'r') as edge_file:
        if edge_list_header:
            next(edge_file) # Skip the header

        for line in edge_file:
            edge = line.strip().split(edge_list_separator)
            src, dst = edge[edge_src_col], edge[edge_dst_col]
            
            # Check if weights are provided
            if edge_weight_col is not None:
                weight = float(edge[edge_weight_col])
            else:
                weight = None
            
            if edge_type_col is not None:
                edge_type = edge[edge_type_col]
            else:
                edge_type = None
            
            if edge_type is None or edge_type == edge_type_name:
                if src is not None and dst is not None:
                    if edge_list_numeric_node_ids:
                        src, dst = int(src), int(dst)
                    if weight is not None:
                        G.add_edge(src, dst, weight=weight)
                    else:
                        G.add_edge(src, dst)
    G.name = name
    
    return G