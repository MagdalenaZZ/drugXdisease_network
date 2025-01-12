import networkx as nx

def compute_graph_features(graph):
    """
    Compute graph-based features such as degree, betweenness, and clustering.

    Parameters:
    - graph: networkx.Graph, the input graph.

    Returns:
    - features: dict, a dictionary of graph features for each node.
    """
    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph)
    clustering = nx.clustering(graph)

    features = {}
    for node in graph.nodes():
        features[node] = {
            "degree": degree[node],
            "betweenness": betweenness[node],
            "clustering": clustering[node],
        }
    return features


def enhanced_features(graph, drug, disease):
    """
    Compute enhanced features for a drug-disease pair, such as shortest path length
    and number of shared neighbors (e.g., genes).

    Parameters:
    - graph: networkx.Graph, the input graph.
    - drug: str, drug node.
    - disease: str, disease node.

    Returns:
    - features: list, enhanced features.
    """
    # Shortest path length
    try:
        shortest_path = nx.shortest_path_length(graph, source=drug, target=disease)
    except nx.NetworkXNoPath:
        shortest_path = -1  # No path exists

    # Number of shared neighbors (genes)
    shared_neighbors = len(set(graph.neighbors(drug)) & set(graph.neighbors(disease)))

    return [shortest_path, shared_neighbors]
