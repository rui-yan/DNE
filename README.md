# DNE

### Deep representation learning of biological networks for enhanced pattern discovery

**TL;DR:** Python implementation of DNE proposed in [our paper](). 
- The proposed method, referred to as discriminative network embedding (DNE), characterizes a node in the network both locally and globally by harnessing the contrast between representations from neighboring and distant nodes.
- By differentiating between local structural patterns and non-local segment patterns, DNE significantly improves node embeddings to more accurately capture both the global structure—such as node structural roles and community memberships—and the local neighborhood connections of each node.
- DNE substantially outperforms existing techniques across various critical biological network analyses, including the inference of biological interactions and the identification of functional modules.
- DNE uniquely improves network embedding by incorporating protein sequence features from pre-trained protein language models, resulting in a substantial enhancement in PPI prediction accuracy.

## Table of Contents
- [Set up the Conda Environment](#set-up-the-conda-environment)
- [Unsupervised pre-training on network data using DNE](#unsupervised-pre-training-on-network-data-using-dne)
- [Downstream analysis with learned node embeddings](#downstream-analysis-with-learned-node-embeddings)
- [Example Jupyter notebooks for using DNE](#example-jupter-notebooks-for-using-dne)

## Set Up the Conda Environment
```bash
git clone https://github.com/rui-yan/DNE.git
cd DNE
conda env create -f dne_conda.yml
conda activate dne
```

* NVIDIA GPU (Tested on Nvidia Quadro RTX 8000 48G x 1) on local workstations
* Python (3.9.18), tensorflow (2.15.0), numpy (1.23.1), pandas (2.2.2), scikit-learn (1.3.0), scipy (1.10.1), seaborn (0.13.2); For further details on the software and package versions used, please refer to the `dne_conda.yml` file.

## Unsupervised pre-training on network data using DNE
```python
from models.dne import DNE
from utils.walker import RandomWalker

# Load network data
graph_data = GraphDataset(data_path)
graph_data.load_graph(dataset)
graph = graph_data.graph

# Initialize a RandomWalker for sampling neighboring nodes using random walks
walker = RandomWalker(graph, p=1.0, q=1.0, use_rejection_sampling=True)
walker.preprocess_transition_probs()
walks = walker.simulate_walks(num_walks=50, walk_length=10)

# Build and train a DNE model using the simulated walks
model = DNE(graph,
            hidden_dim=128,
            pos_embed='LE',
            metric='l1',
            out_activation='sigmoid',
            hidden_dropout_prob=0.1,
            walks=walks)
model.train(batch_size=3000, epochs=5)

# Extract embeddings from the pre-trained DNE model
embeddings = model.get_embeddings()
```

## Downstream analysis with learned node embeddings
The code for performing various downstream tasks using the learned embeddings is available at [code/tasks](https://github.com/rui-yan/DNE/tree/main/code/tasks).

## Example Jupyter notebooks for using DNE 
- Link prediction: Please see our example notebook in [demo.ipynb](https://github.com/rui-yan/DNE/blob/main/demo.ipynb).
