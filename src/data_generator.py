"""
Synthetic fraud network data generator.

Creates realistic transaction graphs with embedded fraud ring patterns,
layered structures, and configurable noise levels for benchmarking
fraud detection algorithms.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FraudNetworkGenerator:
    """Generate synthetic transaction networks with fraud patterns."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.graph_config = self.config.get("graph", {})
        self.rng = np.random.RandomState(
            self.graph_config.get("seed", 42)
        )

    def generate(self) -> Tuple[nx.Graph, Dict[str, int], pd.DataFrame]:
        """Generate a complete fraud network with graph, labels, and transactions.

        Returns:
            graph: NetworkX graph with weighted edges
            labels: Dict mapping node_id to 0 (legitimate) or 1 (fraud)
            transactions: DataFrame of individual transactions
        """
        num_nodes = self.graph_config.get("num_nodes", 3000)
        avg_edges = self.graph_config.get("avg_edges_per_node", 5)
        fraud_rate = self.graph_config.get("fraud_rate", 0.05)

        graph, labels = self._build_base_graph(num_nodes, avg_edges)
        graph, labels = self._inject_fraud_rings(graph, labels)
        graph, labels = self._inject_individual_fraud(
            graph, labels, target_rate=fraud_rate
        )
        transactions = self._generate_transactions(graph, labels)

        n_fraud = sum(labels.values())
        logger.info(
            f"Generated network: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges, {n_fraud} fraud nodes, "
            f"{len(transactions)} transactions"
        )
        return graph, labels, transactions

    def _build_base_graph(
        self, num_nodes: int, avg_edges: int
    ) -> Tuple[nx.Graph, Dict[str, int]]:
        """Create the base legitimate transaction network."""
        G = nx.Graph()
        labels = {}

        for i in range(num_nodes):
            node_id = f"account_{i}"
            G.add_node(
                node_id,
                account_age_days=int(self.rng.exponential(365)),
                avg_transaction_amount=float(
                    self.rng.lognormal(mean=3.5, sigma=1.2)
                ),
                num_counterparties=int(self.rng.poisson(avg_edges)),
                region=self.rng.choice(
                    ["north", "south", "east", "west"]
                ),
            )
            labels[node_id] = 0

        degree_sequence = self.rng.poisson(avg_edges, size=num_nodes)
        degree_sequence = np.clip(degree_sequence, 1, num_nodes - 1)

        nodes = list(G.nodes())
        stubs = []
        for i, deg in enumerate(degree_sequence):
            stubs.extend([nodes[i]] * deg)

        self.rng.shuffle(stubs)
        if len(stubs) % 2 != 0:
            stubs.append(self.rng.choice(nodes))

        for i in range(0, len(stubs) - 1, 2):
            u, v = stubs[i], stubs[i + 1]
            if u != v and not G.has_edge(u, v):
                weight = float(self.rng.lognormal(mean=3.0, sigma=1.5))
                G.add_edge(u, v, weight=weight, edge_type="normal")

        return G, labels

    def _inject_fraud_rings(
        self,
        G: nx.Graph,
        labels: Dict[str, int],
    ) -> Tuple[nx.Graph, Dict[str, int]]:
        """Add densely connected fraud ring substructures."""
        num_rings = self.graph_config.get("num_fraud_rings", 15)
        ring_range = self.graph_config.get("ring_size_range", [3, 12])
        nodes = list(G.nodes())

        for ring_id in range(num_rings):
            ring_size = self.rng.randint(ring_range[0], ring_range[1] + 1)
            ring_members = self.rng.choice(
                nodes, size=ring_size, replace=False
            ).tolist()

            for member in ring_members:
                labels[member] = 1

            ring_type = self.rng.choice(["clique", "cycle", "star", "chain"])

            if ring_type == "clique":
                for i in range(len(ring_members)):
                    for j in range(i + 1, len(ring_members)):
                        if self.rng.random() < 0.8:
                            w = float(self.rng.lognormal(4.0, 0.8))
                            G.add_edge(
                                ring_members[i], ring_members[j],
                                weight=w, edge_type="fraud_ring",
                            )

            elif ring_type == "cycle":
                for i in range(len(ring_members)):
                    j = (i + 1) % len(ring_members)
                    w = float(self.rng.lognormal(4.0, 0.8))
                    G.add_edge(
                        ring_members[i], ring_members[j],
                        weight=w, edge_type="fraud_ring",
                    )
                for _ in range(ring_size // 3):
                    a, b = self.rng.choice(ring_members, 2, replace=False)
                    w = float(self.rng.lognormal(3.5, 1.0))
                    G.add_edge(a, b, weight=w, edge_type="fraud_ring")

            elif ring_type == "star":
                hub = ring_members[0]
                for spoke in ring_members[1:]:
                    w = float(self.rng.lognormal(4.5, 0.5))
                    G.add_edge(
                        hub, spoke, weight=w, edge_type="fraud_ring"
                    )

            elif ring_type == "chain":
                for i in range(len(ring_members) - 1):
                    w = float(self.rng.lognormal(4.0, 0.8))
                    G.add_edge(
                        ring_members[i], ring_members[i + 1],
                        weight=w, edge_type="fraud_ring",
                    )

            if ring_id > 0 and self.rng.random() < 0.25:
                bridge_node = self.rng.choice(ring_members)
                other_fraud = [
                    n for n, l in labels.items()
                    if l == 1 and n not in ring_members
                ]
                if other_fraud:
                    target = self.rng.choice(other_fraud)
                    w = float(self.rng.lognormal(3.0, 1.0))
                    G.add_edge(
                        bridge_node, target,
                        weight=w, edge_type="bridge",
                    )

        return G, labels

    def _inject_individual_fraud(
        self,
        G: nx.Graph,
        labels: Dict[str, int],
        target_rate: float = 0.05,
    ) -> Tuple[nx.Graph, Dict[str, int]]:
        """Add isolated fraudulent nodes (not part of rings)."""
        current_fraud = sum(labels.values())
        target_fraud = int(len(labels) * target_rate)
        additional = max(0, target_fraud - current_fraud)

        if additional > 0:
            legitimate_nodes = [n for n, l in labels.items() if l == 0]
            selected = self.rng.choice(
                legitimate_nodes,
                size=min(additional, len(legitimate_nodes)),
                replace=False,
            )

            for node in selected:
                labels[node] = 1
                for neighbor in list(G.neighbors(node)):
                    if G.has_edge(node, neighbor):
                        current_weight = G[node][neighbor].get("weight", 1.0)
                        G[node][neighbor]["weight"] = (
                            current_weight * self.rng.uniform(2.0, 5.0)
                        )

        return G, labels

    def _generate_transactions(
        self,
        G: nx.Graph,
        labels: Dict[str, int],
    ) -> pd.DataFrame:
        """Generate individual transaction records from the graph edges."""
        records = []
        txn_id = 0

        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            edge_type = data.get("edge_type", "normal")

            if edge_type in ("fraud_ring", "bridge"):
                num_txns = self.rng.randint(3, 15)
            else:
                num_txns = self.rng.randint(1, 5)

            for _ in range(num_txns):
                if self.rng.random() < 0.5:
                    sender, receiver = u, v
                else:
                    sender, receiver = v, u

                amount = float(
                    self.rng.lognormal(
                        mean=np.log(weight), sigma=0.3
                    )
                )
                hour = int(self.rng.choice(24))
                day = int(self.rng.choice(7))
                is_fraud = int(
                    labels.get(sender, 0) == 1 or labels.get(receiver, 0) == 1
                )

                records.append({
                    "transaction_id": f"txn_{txn_id}",
                    "sender_id": sender,
                    "receiver_id": receiver,
                    "amount": round(amount, 2),
                    "hour_of_day": hour,
                    "day_of_week": day,
                    "edge_type": edge_type,
                    "is_fraud": is_fraud,
                })
                txn_id += 1

        df = pd.DataFrame(records)
        return df

    def save_data(
        self,
        graph: nx.Graph,
        labels: Dict[str, int],
        transactions: pd.DataFrame,
        output_dir: str = "data",
    ) -> None:
        """Persist generated data to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        transactions.to_csv(out / "transactions.csv", index=False)

        labels_df = pd.DataFrame([
            {"node_id": k, "is_fraud": v} for k, v in labels.items()
        ])
        labels_df.to_csv(out / "labels.csv", index=False)

        edges = []
        for u, v, d in graph.edges(data=True):
            edges.append({"source": u, "target": v, **d})
        pd.DataFrame(edges).to_csv(out / "edges.csv", index=False)

        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_fraud": sum(labels.values()),
            "fraud_rate": sum(labels.values()) / len(labels),
            "num_transactions": len(transactions),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        }
        with open(out / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Data saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud network"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generator = FraudNetworkGenerator(config_path=args.config)
    graph, labels, transactions = generator.generate()
    generator.save_data(graph, labels, transactions, output_dir=args.output)


if __name__ == "__main__":
    main()
