# Node2Vec for Fraud Detection

Graph embedding approach using Node2Vec random walks to generate node representations from transaction networks. Feeds embeddings into traditional ML classifiers (XGBoost, Random Forest) for fraud detection. A simpler alternative to GNN-based approaches that decouples graph representation learning from classification.

## Overview

This project applies Node2Vec to learn low-dimensional representations of nodes in a transaction graph, then uses these embeddings as features for fraud classification. The approach works in two stages: first, Node2Vec performs biased random walks on the graph to capture structural equivalence and homophily patterns; second, the resulting embeddings are combined with tabular transaction features and fed into XGBoost or Random Forest classifiers.

## Alternative Approach Note

This is an alternative to the GAT (Graph Attention Network) approach for fraud ring detection. Node2Vec generates static graph embeddings using random walks, which can then feed any traditional ML classifier. GAT was chosen for the primary project because it provides end-to-end learning (jointly optimizing graph representation and classification), and handles heterogeneous edge types natively. However, Node2Vec offers simpler deployment since the embeddings can be precomputed and the classifier is a standard scikit-learn/XGBoost model.

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

## Evaluation

Uses a generated synthetic transaction graph for demonstration. No results are published in this repository. Running `python -m src.classifier --config configs/config.yaml` trains the classifier on the synthetic graph and logs precision, recall, F1 and AUC on a held-out split, plus a feature-importance plot.

Note: `FraudClassifier.compare_with_gnn()` falls back to fixed placeholder GNN values when no measured GNN metrics are passed in, so its comparison output is not a real benchmark unless you supply metrics from an actual GAT run.

## License

MIT License
