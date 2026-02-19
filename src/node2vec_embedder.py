"""
Node2Vec embedding generation for transaction graphs.

Trains Node2Vec models with configurable random walk parameters (p, q)
to learn low-dimensional node representations that capture both
structural equivalence and community membership patterns.
"""

import argparse
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class Node2VecEmbedder:
    """Train and manage Node2Vec graph embeddings."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.n2v_config = self.config.get("node2vec", {})
        self.dimensions = self.n2v_config.get("dimensions", 64)
        self.walk_length = self.n2v_config.get("walk_length", 30)
        self.num_walks = self.n2v_config.get("num_walks", 200)
        self.p = self.n2v_config.get("p", 1.0)
        self.q = self.n2v_config.get("q", 0.5)
        self.window = self.n2v_config.get("window", 10)
        self.min_count = self.n2v_config.get("min_count", 1)
        self.workers = self.n2v_config.get("workers", 4)
        self.seed = self.n2v_config.get("seed", 42)

        self._node2vec_model: Optional[Node2Vec] = None
        self._w2v_model: Optional[Word2Vec] = None
        self._graph: Optional[nx.Graph] = None

    def fit(
        self,
        graph: nx.Graph,
        p: Optional[float] = None,
        q: Optional[float] = None,
        dimensions: Optional[int] = None,
        walk_length: Optional[int] = None,
        num_walks: Optional[int] = None,
    ) -> "Node2VecEmbedder":
        """Train Node2Vec model on the given graph.

        Performs biased random walks parameterized by p and q, then
        trains a Word2Vec skip-gram model on the walk sequences to
        learn node embeddings.
        """
        self._graph = graph

        _p = p or self.p
        _q = q or self.q
        _dim = dimensions or self.dimensions
        _wl = walk_length or self.walk_length
        _nw = num_walks or self.num_walks

        logger.info(
            f"Training Node2Vec: dims={_dim}, walks={_nw}, "
            f"walk_length={_wl}, p={_p}, q={_q}"
        )

        self._node2vec_model = Node2Vec(
            graph,
            dimensions=_dim,
            walk_length=_wl,
            num_walks=_nw,
            p=_p,
            q=_q,
            workers=self.workers,
            seed=self.seed,
            quiet=True,
        )

        self._w2v_model = self._node2vec_model.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=4,
            seed=self.seed,
        )

        vocab_size = len(self._w2v_model.wv)
        logger.info(
            f"Node2Vec training complete: {vocab_size} node embeddings "
            f"of dimension {_dim}"
        )
        return self

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Return a dictionary mapping node IDs to embedding vectors."""
        if self._w2v_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        embeddings = {}
        for node in self._w2v_model.wv.index_to_key:
            embeddings[node] = self._w2v_model.wv[node]

        return embeddings

    def get_embedding_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Return ordered node IDs and the corresponding embedding matrix."""
        embeddings = self.get_embeddings()
        nodes = sorted(embeddings.keys())
        matrix = np.array([embeddings[n] for n in nodes])
        return nodes, matrix

    def get_node_embedding(self, node_id: str) -> np.ndarray:
        """Retrieve the embedding vector for a single node."""
        if self._w2v_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if node_id not in self._w2v_model.wv:
            raise KeyError(f"Node '{node_id}' not found in embedding vocabulary")

        return self._w2v_model.wv[node_id]

    def similar_nodes(
        self, node_id: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find the most similar nodes by embedding cosine distance."""
        if self._w2v_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if node_id not in self._w2v_model.wv:
            raise KeyError(f"Node '{node_id}' not found in vocabulary")

        similar = self._w2v_model.wv.most_similar(node_id, topn=top_k)
        return similar

    def visualize_embeddings(
        self,
        labels: Optional[Dict[str, int]] = None,
        method: str = "tsne",
        perplexity: int = 30,
        max_nodes: int = 2000,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create a 2D visualization of node embeddings using t-SNE or UMAP.

        If labels are provided, colors points by fraud/legitimate status.
        """
        nodes, matrix = self.get_embedding_matrix()

        if len(nodes) > max_nodes:
            rng = np.random.RandomState(self.seed)
            indices = rng.choice(len(nodes), max_nodes, replace=False)
            nodes = [nodes[i] for i in indices]
            matrix = matrix[indices]

        if method == "tsne":
            perp = min(perplexity, len(nodes) - 1)
            reducer = TSNE(
                n_components=2,
                perplexity=perp,
                random_state=self.seed,
                n_iter=1000,
            )
            coords = reducer.fit_transform(matrix)
        else:
            try:
                from umap import UMAP
                reducer = UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    random_state=self.seed,
                )
                coords = reducer.fit_transform(matrix)
            except ImportError:
                logger.warning("UMAP not installed, falling back to t-SNE")
                reducer = TSNE(
                    n_components=2,
                    perplexity=min(perplexity, len(nodes) - 1),
                    random_state=self.seed,
                )
                coords = reducer.fit_transform(matrix)

        fig, ax = plt.subplots(figsize=(12, 10))

        if labels is not None:
            colors = []
            for node in nodes:
                if labels.get(node, 0) == 1:
                    colors.append("#E53935")
                else:
                    colors.append("#42A5F5")

            fraud_mask = np.array([labels.get(n, 0) == 1 for n in nodes])
            legit_mask = ~fraud_mask

            ax.scatter(
                coords[legit_mask, 0], coords[legit_mask, 1],
                c="#42A5F5", alpha=0.4, s=10, label="Legitimate",
            )
            ax.scatter(
                coords[fraud_mask, 0], coords[fraud_mask, 1],
                c="#E53935", alpha=0.8, s=30, label="Fraud",
                edgecolors="black", linewidths=0.5,
            )
            ax.legend(fontsize=12)
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c="#42A5F5", alpha=0.5, s=15,
            )

        method_name = method.upper()
        ax.set_title(
            f"Node2Vec Embeddings ({method_name} Projection)", fontsize=14
        )
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Embedding visualization saved to {save_path}")

        return fig

    def save_embeddings(self, path: str = "embeddings/node2vec.model") -> None:
        """Persist the trained Word2Vec model to disk."""
        if self._w2v_model is None:
            raise RuntimeError("No trained model to save.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._w2v_model.save(str(output_path))
        logger.info(f"Embeddings saved to {path}")

    def load_embeddings(self, path: str = "embeddings/node2vec.model") -> None:
        """Load a previously trained Word2Vec model from disk."""
        self._w2v_model = Word2Vec.load(path)
        logger.info(
            f"Loaded embeddings from {path}: "
            f"{len(self._w2v_model.wv)} nodes"
        )

    def grid_search_params(
        self,
        graph: nx.Graph,
        labels: Dict[str, int],
        p_values: Optional[List[float]] = None,
        q_values: Optional[List[float]] = None,
        dim_values: Optional[List[int]] = None,
        walk_lengths: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Search over Node2Vec hyperparameters to find the best configuration.

        Evaluates each combination by training embeddings and measuring
        how well a simple logistic regression separates fraud from
        legitimate nodes using the embeddings.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score

        p_values = p_values or [0.5, 1.0, 2.0]
        q_values = q_values or [0.25, 0.5, 1.0, 2.0]
        dim_values = dim_values or [32, 64, 128]
        walk_lengths = walk_lengths or [20, 30]

        param_grid = list(
            itertools.product(p_values, q_values, dim_values, walk_lengths)
        )
        logger.info(
            f"Grid search over {len(param_grid)} parameter combinations"
        )

        results = []
        for p, q, dim, wl in param_grid:
            try:
                self.fit(graph, p=p, q=q, dimensions=dim, walk_length=wl)
                embeddings = self.get_embeddings()

                nodes_with_labels = [
                    n for n in embeddings.keys() if n in labels
                ]
                X = np.array([embeddings[n] for n in nodes_with_labels])
                y = np.array([labels[n] for n in nodes_with_labels])

                if len(np.unique(y)) < 2:
                    continue

                clf = LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=self.seed,
                )
                scores = cross_val_score(
                    clf, X, y, cv=3, scoring="roc_auc"
                )
                mean_auc = float(np.mean(scores))

                result = {
                    "p": p,
                    "q": q,
                    "dimensions": dim,
                    "walk_length": wl,
                    "mean_auc": round(mean_auc, 4),
                    "std_auc": round(float(np.std(scores)), 4),
                }
                results.append(result)
                logger.info(
                    f"p={p}, q={q}, dim={dim}, wl={wl} -> AUC={mean_auc:.4f}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed for p={p}, q={q}, dim={dim}, wl={wl}: {e}"
                )

        results.sort(key=lambda x: x["mean_auc"], reverse=True)

        if results:
            best = results[0]
            logger.info(
                f"Best config: p={best['p']}, q={best['q']}, "
                f"dim={best['dimensions']}, wl={best['walk_length']} "
                f"(AUC={best['mean_auc']:.4f})"
            )

        return results


def main():
    parser = argparse.ArgumentParser(description="Node2Vec embeddings")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="embeddings/node2vec.model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from src.graph_builder import TransactionGraphBuilder

    builder = TransactionGraphBuilder(config_path=args.config)
    graph, labels = builder.generate_synthetic_graph()

    embedder = Node2VecEmbedder(config_path=args.config)
    embedder.fit(graph)
    embedder.save_embeddings(args.output)

    embedder.visualize_embeddings(
        labels=labels, save_path="embedding_viz.png"
    )


if __name__ == "__main__":
    main()
