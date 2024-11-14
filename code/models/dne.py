import numpy as np
import torch
import torch.nn as nn
from .le import LE
import sys
sys.path.append("..")
from utils.utils_graph import preprocess_nxgraph
from utils.walker import RandomWalker
from tqdm import tqdm
from collections import defaultdict
import networkx as nx

class MLP(nn.Module):
    """backbone"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(MLP, self).__init__()
        self.mlp_layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                    )
    
    def forward(self, x) :
        z = self.mlp_layers(x)
        return z


class Similarity(nn.Module):
    def __init__(self, input_dim, out_dim=1, activation_fn=None):
        super(Similarity, self).__init__()
        self.dense = nn.Linear(input_dim, out_dim)
        self.activation_fn = activation_fn
    
    def forward(self, x):
        # x is expected to be of shape [num_pairs, 2, feature_dim]
        l1_distance = torch.abs(x[:, 0, :] - x[:, 1, :])
        output = self.dense(l1_distance)
        if self.activation_fn:
            output = self.activation_fn(output)
        return output


class GraphMLP(nn.Module):
    def __init__(self, x_pos_dim, x_dim=None, hidden_dim=128, dropout=0.3):
        super(GraphMLP, self).__init__()
        
        self.use_dual_encoding = x_dim is not None
        
        # adjust hidden dimensions based on the presence of x_dim
        hidden_dim_pos = hidden_dim // 2 if self.use_dual_encoding else hidden_dim
        
        # define encoder for node positional features
        self.encoder_pos = MLP(x_pos_dim, hidden_dim_pos, dropout)
        
        # define encoder for node features (optional)
        self.encoder_feat = MLP(x_dim, hidden_dim // 2, dropout) if self.use_dual_encoding else None
        
        # define similarity function
        self.similarity = Similarity(hidden_dim, activation_fn=torch.sigmoid)

    def reset_parameters(self):
        """Reset parameters of all sub-modules."""
        self.encoder_pos.reset_parameters()
        if self.encoder_feat:
            self.encoder_feat.reset_parameters()
    
    def dual_encoder(self, x_pos, x_feat=None):
        """Encode inputs using dual encoders, if available."""
        z_pos = self.encoder_pos(x_pos)
        z_feat = self.encoder_feat(x_feat) if self.encoder_feat and x_feat is not None else None
        
        if z_feat is not None:
            z = torch.cat((z_pos, z_feat), dim=1)
        else:
            z = z_pos
        
        return z
    
    def forward(self, x_pos_1, x_pos_2, x_feat_1=None, x_feat_2=None):
        """Forward pass through the graph MLP model."""
        h_1 = self.dual_encoder(x_pos_1, x_feat_1)
        h_2 = self.dual_encoder(x_pos_2, x_feat_2)
        h = torch.stack((h_1, h_2), dim=1)
        return self.similarity(h)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        square_pred = torch.square(y_pred)
        margin_square = torch.square(torch.clamp(self.margin - y_pred, min=0))
        loss_contrastive = torch.mean((1 - y_true) * square_pred + y_true * margin_square)
        return loss_contrastive


class DNE:
    def __init__(self, 
                 graph, 
                 feat=None,
                 hidden_dim=128, 
                 num_pos_features=256, 
                 pos_embed='LE',
                 metric='l1', 
                 out_activation='sigmoid', 
                 hidden_dropout_prob=0.1, walks=None):
        super().__init__()
        
        self.feat = feat
        self.walks = walks
        
        self.hidden_dim = hidden_dim
        self.num_pos_features = num_pos_features
        self.num_features = feat.shape[1] if feat is not None else None

        self.pos_embed = pos_embed
        self.hidden_dropout_prob = hidden_dropout_prob
        self.metric = metric
        self.out_activation = out_activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.graph = nx.relabel_nodes(graph, self.node2idx)
        
        self._embeddings = {}
    
    def _simulate_walks(self, n, l, p, q, use_rejection_sampling=True):
        walker = RandomWalker(self.graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)
        walker.preprocess_transition_probs()
        walks = walker.simulate_walks(num_walks=n, walk_length=l, workers=1, verbose=1)
        return walks
    
    def prepare_data(self, n, l, p, q):
        # Simulate random walks
        walks = self._simulate_walks(n=n, l=l, p=p, q=q)
        
        degrees = defaultdict(int, dict(self.graph.degree()))
        all_nodes = list(degrees.keys())

        sampling_distribution = np.array([degrees[n] ** 0.75 for n in all_nodes], dtype=np.float32)
        sampling_distribution_norm = sampling_distribution / sampling_distribution.sum()

        targets = [walk[0] for walk in walks]
        positive_pairs = torch.tensor(
            [(target, context) for target, walk in zip(targets, walks) for context in walk[1:]],
            dtype=torch.long
        )

        negative_samples = np.random.choice(all_nodes, size=len(positive_pairs), p=sampling_distribution_norm)
        negative_pairs = torch.stack((positive_pairs[:, 0], torch.from_numpy(negative_samples)), dim=1)

        # Combine positive and negative pairs and generate labels
        node_pairs = torch.cat((positive_pairs, negative_pairs), dim=0)
        labels = torch.cat((torch.ones(len(positive_pairs)), torch.zeros(len(negative_pairs)))).long()

        # Shuffle pairs and labels
        indices = torch.randperm(len(node_pairs))
        node_pairs = node_pairs[indices]
        labels = labels[indices]
        
        return node_pairs, labels

    def compute_pos_emb(self, num_pos_features=256):
        pos_emb_model = LE(self.graph, num_pos_features)
        x_pos = pos_emb_model._X
        return x_pos

    def train(self, batch_size=1000, epochs=5, walk_number=50, walk_length=10, p=1.0, q=1.0):
        
        self.x_pos = self.compute_pos_emb(num_pos_features=self.num_pos_features)
        x_pos = torch.from_numpy(self.x_pos).float().to(self.device)
        
        if self.num_features is not None:
            x = torch.from_numpy(self.feat).float().to(self.device)

        node_pairs, labels = self.prepare_data(walk_number, walk_length, p, q)   
        node_pairs = node_pairs.to(self.device)
        labels = labels.to(self.device)
        
        model = GraphMLP(self.num_pos_features, self.num_features, self.hidden_dim).to(self.device)
        criterion = ContrastiveLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        n_batch = (len(node_pairs) + batch_size - 1) // batch_size
        
        # Training loop
        epoch_iter = tqdm(range(epochs), desc="Training Epochs")
        for epoch in epoch_iter:
            model.train()
            total_loss = 0.0

            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(node_pairs))
                batch_node_pairs = node_pairs[start:end]
                batch_labels = labels[start:end]
            
                optimizer.zero_grad()
                
                if self.num_features is not None:
                    out = model(
                        x_pos[batch_node_pairs[:, 0]], x_pos[batch_node_pairs[:, 1]],
                        x[batch_node_pairs[:, 0]], x[batch_node_pairs[:, 1]]
                    )
                else:
                    out = model(x_pos[batch_node_pairs[:, 0]], x_pos[batch_node_pairs[:, 1]])
                
                # Calculate loss and backpropagate
                out = out.squeeze()
                loss = criterion(batch_labels, out)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            # Calculate average loss for the epoch
            avg_loss = total_loss / n_batch

            epoch_iter.set_description(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        self.model = model

    def get_embeddings(self):
        self._embeddings = {}
        self.model.eval()
        
        x_pos = torch.from_numpy(self.x_pos).float().to(self.device)
        
        if self.num_features is not None:
            x = torch.from_numpy(self.feat).float().to(self.device)
            embeddings = self.model.dual_encoder(x_pos, x)
        else:
            embeddings = self.model.dual_encoder(x_pos)
        
        embeddings = embeddings.detach().cpu().numpy()
        idx2node = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[idx2node[i]] = embedding
        
        return self._embeddings