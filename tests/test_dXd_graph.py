import unittest
from drugXdisease.dXd_graph import sample_graph

class TestSampleGraph(unittest.TestCase):
    def test_graph_structure(self):
        num_drugs = 10
        num_diseases = 5
        num_genes = 3
        num_edges = 15
        treats_ratio = 0.4
        graph = sample_graph(num_drugs, num_diseases, num_genes, num_edges, treats_ratio)

        # Count additional edges
        drug_gene_edges = num_drugs  # Each drug has at least one gene target
        disease_gene_edges = num_diseases  # One disease-gene relationship per disease
        similarity_edges = sum(1 for u, v in graph.edges if graph[u][v].get("relation") == "similar")

        # Calculate total expected edges
        expected_edges = num_edges + drug_gene_edges + disease_gene_edges + similarity_edges

        # Check the number of nodes and edges
        self.assertEqual(len(graph.nodes), num_drugs + num_diseases + num_genes)
        self.assertEqual(len(graph.edges), expected_edges)

        # Check edge labels
        treats_count = sum(1 for u, v, d in graph.edges(data=True) if d.get("relation") == "treats")
        self.assertAlmostEqual(treats_count / num_edges, treats_ratio, delta=0.1)


