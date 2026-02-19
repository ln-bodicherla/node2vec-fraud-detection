"""
Transaction graph construction for fraud detection.

Builds weighted graphs from transaction data where nodes represent
accounts/entities and edges represent financial interactions. Supports
bipartite provider-patient graphs and account-level transaction graphs
with engineered graph-structural features.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class TransactionGraphBuilder:
    """Constructs and analyzes transaction graphs for fraud detection."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.graph_config = self.config.get("graph", {})
        self.graph: Optional[nx.Graph] = None
        self.node_labels: Dict[str, int] = {}

    def build_bipartite_graph(
        self,
        transactions: pd.DataFrame,
        source_col: str = "provider_id",
        target_col: str = "patient_id",
        weight_col: Optional[str] = "amount",
    ) -> nx.Graph:
        """Build a bipartite graph between providers and patients.

        Nodes on each side represent providers and patients respectively.
        Edges are weighted by total transaction amount or frequency.
        """
        G = nx.Graph()

        sources = transactions[source_col].unique()
        targets = transactions[target_col].unique()

        for s in sources:
            G.add_node(s, bipartite=0, node_type="provider")
        for t in targets:
            G.add_node(t, bipartite=1, node_type="patient")

        if weight_col and weight_col in transactions.columns:
            edge_weights = (
                transactions.groupby([source_col, target_col])[weight_col]
                .sum()
                .reset_index()
            )
            for _, row in edge_weights.iterrows():
                G.add_edge(
                    row[source_col],
                    row[target_col],
                    weight=float(row[weight_col]),
                )
        else:
            edge_counts = (
                transactions.groupby([source_col, target_col])
                .size()
                .reset_index(name="count")
            )
            for _, row in edge_counts.iterrows():
                G.add_edge(
                    row[source_col],
                    row[target_col],
                    weight=float(row["count"]),
                )

        self.graph = G
        logger.info(
            f"Built bipartite graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        return G

    def build_transaction_graph(
        self,
        transactions: pd.DataFrame,
        source_col: str = "sender_id",
        target_col: str = "receiver_id",
        weight_col: Optional[str] = "amount",
    ) -> nx.Graph:
        """Build an account-level transaction graph.

        Nodes represent accounts. Edges connect accounts that have
        transacted with each other, weighted by frequency or amount.
        """
        G = nx.Graph()

        all_accounts = set(
            transactions[source_col].tolist()
            + transactions[target_col].tolist()
        )
        for account in all_accounts:
            G.add_node(account)

        if weight_col and weight_col in transactions.columns:
            grouped = (
                transactions.groupby([source_col, target_col])
                .agg(
                    total_amount=(weight_col, "sum"),
                    num_transactions=(weight_col, "count"),
                )
                .reset_index()
            )
            for _, row in grouped.iterrows():
                G.add_edge(
                    row[source_col],
                    row[target_col],
                    weight=float(row["total_amount"]),
                    num_transactions=int(row["num_transactions"]),
                )
        else:
            edge_counts = (
                transactions.groupby([source_col, target_col])
                .size()
                .reset_index(name="count")
            )
            for _, row in edge_counts.iterrows():
                G.add_edge(
                    row[source_col],
                    row[target_col],
                    weight=float(row["count"]),
                    num_transactions=int(row["count"]),
                )

        self.graph = G
        logger.info(
            f"Built transaction graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        return G

    def add_edge_weights(
        self,
        transactions: pd.DataFrame,
        source_col: str = "sender_id",
        target_col: str = "receiver_id",
    ) -> None:
        """Enrich edge weights with transaction frequency and amounts."""
        if self.graph is None:
            raise RuntimeError("No graph built. Call build_*_graph() first.")

        for u, v in self.graph.edges():
            mask = (
                (transactions[source_col] == u) & (transactions[target_col] == v)
            ) | (
                (transactions[source_col] == v) & (transactions[target_col] == u)
            )
            edge_txns = transactions[mask]

            if "amount" in edge_txns.columns:
                self.graph[u][v]["total_amount"] = float(
                    edge_txns["amount"].sum()
                )
                self.graph[u][v]["avg_amount"] = float(
                    edge_txns["amount"].mean()
                )
                self.graph[u][v]["max_amount"] = float(
                    edge_txns["amount"].max()
                )
            self.graph[u][v]["frequency"] = len(edge_txns)

    def compute_graph_features(
        self, G: Optional[nx.Graph] = None
    ) -> pd.DataFrame:
        """Compute structural features for each node.

        Features include degree centrality, betweenness centrality,
        closeness centrality, clustering coefficient, PageRank, and
        local community detection metrics.
        """
        graph = G or self.graph
        if graph is None:
            raise RuntimeError("No graph available.")

        nodes = list(graph.nodes())
        features: Dict[str, Dict[str, float]] = {n: {} for n in nodes}

        degree_centrality = nx.degree_centrality(graph)
        for node in nodes:
            features[node]["degree"] = float(graph.degree(node))
            features[node]["degree_centrality"] = degree_centrality[node]

        weighted_degree = {}
        for node in nodes:
            w_sum = sum(
                d.get("weight", 1.0) for _, _, d in graph.edges(node, data=True)
            )
            weighted_degree[node] = w_sum
            features[node]["weighted_degree"] = w_sum

        try:
            betweenness = nx.betweenness_centrality(graph, k=min(100, len(nodes)))
        except Exception:
            betweenness = {n: 0.0 for n in nodes}

        closeness = nx.closeness_centrality(graph)

        clustering = nx.clustering(graph)

        try:
            pagerank = nx.pagerank(graph, max_iter=100)
        except Exception:
            pagerank = {n: 1.0 / len(nodes) for n in nodes}

        for node in nodes:
            features[node]["betweenness_centrality"] = betweenness.get(node, 0.0)
            features[node]["closeness_centrality"] = closeness.get(node, 0.0)
            features[node]["clustering_coefficient"] = clustering.get(node, 0.0)
            features[node]["pagerank"] = pagerank.get(node, 0.0)

        for node in nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_degrees = [graph.degree(n) for n in neighbors]
                features[node]["avg_neighbor_degree"] = float(
                    np.mean(neighbor_degrees)
                )
                features[node]["max_neighbor_degree"] = float(
                    max(neighbor_degrees)
                )
            else:
                features[node]["avg_neighbor_degree"] = 0.0
                features[node]["max_neighbor_degree"] = 0.0

        try:
            triangles = nx.triangles(graph)
            for node in nodes:
                features[node]["triangles"] = float(triangles.get(node, 0))
        except Exception:
            for node in nodes:
                features[node]["triangles"] = 0.0

        df = pd.DataFrame.from_dict(features, orient="index")
        df.index.name = "node_id"
        df = df.reset_index()

        logger.info(
            f"Computed {len(df.columns) - 1} graph features for "
            f"{len(df)} nodes"
        )
        return df

    def generate_synthetic_graph(
        self, seed: Optional[int] = None
    ) -> Tuple[nx.Graph, Dict[str, int]]:
        """Create a realistic fraud network for demonstration.

        Generates a transaction graph with:
        - Normal transaction patterns (random edges)
        - Fraud rings (densely connected subgraphs)
        - Bridge nodes connecting fraud rings

        Returns the graph and a dict mapping node IDs to labels (0/1).
        """
        cfg = self.graph_config
        num_nodes = cfg.get("num_nodes", 3000)
        avg_edges = cfg.get("avg_edges_per_node", 5)
        fraud_rate = cfg.get("fraud_rate", 0.05)
        num_fraud_rings = cfg.get("num_fraud_rings", 15)
        ring_size_range = cfg.get("ring_size_range", [3, 12])
        weight_range = cfg.get("weight_range", [1.0, 100.0])

        rng = np.random.RandomState(seed or cfg.get("seed", 42))

        G = nx.Graph()
        labels = {}

        for i in range(num_nodes):
            node_id = f"account_{i}"
            G.add_node(node_id)
            labels[node_id] = 0

        num_edges = int(num_nodes * avg_edges / 2)
        node_list = list(G.nodes())
        edges_added = 0
        while edges_added < num_edges:
            u = rng.choice(node_list)
            v = rng.choice(node_list)
            if u != v and not G.has_edge(u, v):
                weight = rng.uniform(weight_range[0], weight_range[1])
                G.add_edge(u, v, weight=weight)
                edges_added += 1

        fraud_nodes = set()
        for ring_id in range(num_fraud_rings):
            ring_size = rng.randint(ring_size_range[0], ring_size_range[1] + 1)
            start_idx = rng.randint(0, num_nodes - ring_size)
            ring_members = [
                f"account_{start_idx + j}" for j in range(ring_size)
            ]

            for j in range(len(ring_members)):
                for k in range(j + 1, len(ring_members)):
                    if rng.random() < 0.7:
                        weight = rng.uniform(
                            weight_range[1] * 0.5, weight_range[1] * 3.0
                        )
                        G.add_edge(
                            ring_members[j], ring_members[k], weight=weight
                        )
                fraud_nodes.add(ring_members[j])

            if ring_id > 0 and rng.random() < 0.3:
                bridge = rng.choice(ring_members)
                prev_ring_start = rng.randint(0, num_nodes)
                prev_node = f"account_{prev_ring_start}"
                G.add_edge(bridge, prev_node, weight=rng.uniform(50, 200))

        for node in fraud_nodes:
            labels[node] = 1

        additional_fraud = int(num_nodes * fraud_rate) - len(fraud_nodes)
        if additional_fraud > 0:
            non_fraud = [n for n in node_list if labels[n] == 0]
            extra = rng.choice(non_fraud, size=min(additional_fraud, len(non_fraud)), replace=False)
            for node in extra:
                labels[node] = 1
                neighbors = list(G.neighbors(node))
                for nbr in neighbors[:3]:
                    G[node][nbr]["weight"] *= rng.uniform(2.0, 5.0)

        self.graph = G
        self.node_labels = labels
        n_fraud = sum(labels.values())
        logger.info(
            f"Generated synthetic graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, {n_fraud} fraud nodes "
            f"({n_fraud / len(labels) * 100:.1f}%)"
        )
        return G, labels

    def visualize_network(
        self,
        G: Optional[nx.Graph] = None,
        labels: Optional[Dict[str, int]] = None,
        max_nodes: int = 500,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize the transaction graph with fraud nodes highlighted.

        Uses a spring layout and colors nodes by fraud label. Large
        graphs are subsampled for readability.
        """
        graph = G or self.graph
        node_labels = labels or self.node_labels

        if graph is None:
            raise RuntimeError("No graph available.")

        if graph.number_of_nodes() > max_nodes:
            fraud_nodes = [n for n, l in node_labels.items() if l == 1]
            non_fraud = [n for n, l in node_labels.items() if l == 0]
            rng = np.random.RandomState(42)
            sampled_non_fraud = rng.choice(
                non_fraud,
                size=min(max_nodes - len(fraud_nodes), len(non_fraud)),
                replace=False,
            ).tolist()
            keep_nodes = set(fraud_nodes + sampled_non_fraud)
            graph = graph.subgraph(keep_nodes).copy()

        fig, ax = plt.subplots(figsize=(14, 10))

        pos = nx.spring_layout(graph, seed=42, k=1.5 / np.sqrt(graph.number_of_nodes()))

        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            if node_labels.get(node, 0) == 1:
                node_colors.append("#E53935")
                node_sizes.append(60)
            else:
                node_colors.append("#90CAF9")
                node_sizes.append(20)

        nx.draw_networkx_edges(
            graph, pos, alpha=0.1, edge_color="#BDBDBD", ax=ax
        )
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax,
        )

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935",
                   markersize=10, label="Fraud"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9",
                   markersize=10, label="Legitimate"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

        n_fraud = sum(1 for n in graph.nodes() if node_labels.get(n, 0) == 1)
        ax.set_title(
            f"Transaction Network ({graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, {n_fraud} fraud)",
            fontsize=14,
        )
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Network visualization saved to {save_path}")

        return fig
