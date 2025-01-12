import random
import networkx as nx

def sample_graph(num_drugs, num_diseases, num_genes, num_edges, treats_ratio):
    """
    Create a sample graph with specified numbers of drugs, diseases, genes, and edges.
    """
    G = nx.Graph()
    drugs = [f"drug{i}" for i in range(1, num_drugs + 1)]
    diseases = [f"disease{i}" for i in range(1, num_diseases + 1)]
    genes = [f"gene{i}" for i in range(1, num_genes + 1)]

    G.add_nodes_from(drugs, type="drug")
    G.add_nodes_from(diseases, type="disease")
    G.add_nodes_from(genes, type="gene")

    num_treats = int(num_edges * treats_ratio)
    num_unaffected = num_edges - num_treats

    edges_treats = [
        (random.choice(drugs), random.choice(diseases), {"relation": "treats"})
        for _ in range(num_treats)
    ]
    edges_unaffected = [
        (random.choice(drugs), random.choice(diseases), {"relation": "unaffected"})
        for _ in range(num_unaffected)
    ]

    G.add_edges_from(edges_treats)
    G.add_edges_from(edges_unaffected)

    drug_gene_edges = []
    for drug in drugs:
        gene_target = random.choice(genes)
        drug_gene_edges.append((drug, gene_target, {"relation": "targets"}))
        if random.random() < 0.1:
            second_gene_target = random.choice(genes)
            drug_gene_edges.append((drug, second_gene_target, {"relation": "targets"}))

    G.add_edges_from(drug_gene_edges)

    disease_gene_edges = [
        (random.choice(diseases), random.choice(genes), {"relation": "associated"})
        for _ in range(num_diseases)
    ]
    G.add_edges_from(disease_gene_edges)

    for disease1 in diseases:
        for disease2 in diseases:
            if disease1 != disease2:
                shared_genes = len(set(G.neighbors(disease1)) & set(G.neighbors(disease2)))
                if shared_genes > 0:
                    G.add_edge(disease1, disease2, relation="similar", weight=shared_genes)

    return G
