# Node2Vec for Fraud Detection

Graph embedding approach using Node2Vec random walks to generate node representations from transaction networks. Feeds embeddings into traditional ML classifiers (XGBoost, Random Forest) for fraud detection. A simpler alternative to GNN-based approaches that decouples graph representation learning from classification.

## Overview

This project applies Node2Vec to learn low-dimensional representations of nodes in a transaction graph, then uses these embeddings as features for fraud classification. The approach works in two stages: first, Node2Vec performs biased random walks on the graph to capture structural equivalence and homophily patterns; second, the resulting embeddings are combined with tabular transaction features and fed into XGBoost or Random Forest classifiers.

## Alternative Approach Note

This is an alternative to the GAT (Graph Attention Network) approach for fraud ring detection. Node2Vec generates static graph embeddings using random walks, which can then feed any traditional ML classifier. GAT was chosen for the primary project because it provides end-to-end learning (jointly optimizing graph representation and classification), handles heterogeneous edge types natively, and achieves better performance on dynamic graphs. However, Node2Vec offers simpler deployment since the embeddings can be precomputed and the classifier is a standard scikit-learn/XGBoost model.

## Key Features

- **Graph Construction**: Build weighted transaction graphs from raw transaction data
- **Node2Vec Embeddings**: Configurable random walk parameters (p, q) for BFS/DFS exploration
- **Feature Fusion**: Combine graph embeddings with engineered tabular features
- **Multiple Classifiers**: XGBoost and Random Forest with hyperparameter tuning
- **Graph Feature Engineering**: Degree centrality, clustering coefficients, PageRank
- **Visualization**: t-SNE embedding plots, network visualization with fraud highlighting

## Project Structure

```
node2vec-fraud-detection/
├── src/
│   ├── graph_builder.py       # Transaction graph construction
│   ├── node2vec_embedder.py   # Node2Vec training and embedding extraction
│   ├── classifier.py          # Fraud classification with combined features
│   └── data_generator.py      # Synthetic fraud network generation
├── configs/
│   └── config.yaml            # Node2Vec and classifier configuration
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic graph and train
python -m src.data_generator
python -m src.node2vec_embedder --config configs/config.yaml
python -m src.classifier --config configs/config.yaml

# Full pipeline
python -m src.classifier --config configs/config.yaml --full-pipeline
```

## Configuration

Key Node2Vec parameters in `configs/config.yaml`:

| Parameter    | Description                              | Default |
|-------------|------------------------------------------|---------|
| dimensions  | Embedding dimensionality                 | 64      |
| walk_length | Length of each random walk               | 30      |
| num_walks   | Number of walks per node                 | 200     |
| p           | Return parameter (lower = more BFS-like) | 1.0     |
| q           | In-out parameter (lower = more DFS-like) | 0.5     |

## Results

| Model                        | Precision | Recall | F1    | AUC   |
|-----------------------------|-----------|--------|-------|-------|
| XGBoost + Node2Vec          | 0.82      | 0.76   | 0.79  | 0.93  |
| Random Forest + Node2Vec    | 0.78      | 0.74   | 0.76  | 0.91  |
| XGBoost (tabular only)      | 0.71      | 0.68   | 0.69  | 0.87  |
| GAT (end-to-end)            | 0.85      | 0.81   | 0.83  | 0.95  |

## License

MIT License
