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
- [Example Jupyter notebooks for using DNE](#example-jupyter-notebooks-for-using-dne)

## Set Up the Conda Environment
```bash
git clone https://github.com/rui-yan/DNE.git
cd DNE
conda env create -f dne_conda.yml
conda activate dne
```

* NVIDIA GPU (Tested on Nvidia Quadro RTX 8000 48G x 1) on local workstations
* Python (3.9.20), torch (2.5.1), torch-geometric (2.6.1), networkx (3.2.1), numpy (1.26.4), pandas (2.2.3), scikit-learn (1.5.2), scipy (1.13.1); For further details on the software and package versions used, please refer to the `dne_conda.yml` file.

## Unsupervised pre-training on network data using DNE
```python
from models.dne import DNE
from dataset import GraphDataset

graph_data = GraphDataset(data_path)
graph_data.load_graph(dataset)
graph = graph_data.graph

# Build and train a DNE model using the simulated walks
model = DNE(graph, hidden_dim=128)
model.train(batch_size=5120, epochs=10)

# Extract embeddings from the pre-trained DNE model
embeddings = model.get_embeddings()
```

## Downstream analysis with learned node embeddings
The code for performing various downstream tasks using the learned embeddings is available at [code/tasks](https://github.com/rui-yan/DNE/tree/main/code/tasks).

## Example Jupyter notebooks for using DNE 
- Link prediction: Please see our example notebook in [demo.ipynb](https://github.com/rui-yan/DNE/blob/main/demo.ipynb).
